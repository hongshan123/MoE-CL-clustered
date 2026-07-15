"""聚类共享 LoRA 池的状态、相似度计算和参数合并策略。"""

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch


LoraPair = Tuple[torch.Tensor, torch.Tensor]
LoraDeltaMap = Dict[str, LoraPair]


@dataclass
class SharedPoolState:
    """共享专家池的可序列化状态，随 adapter checkpoint 一起保存。"""
    mode: str = "single"
    max_shared_experts: int = 3
    probe_steps: int = 100
    similarity_threshold: float = 0.02
    shared_expert_count: int = 1
    task_to_shared: Dict[int, int] = field(default_factory=dict)
    shared_histories: Dict[int, List[int]] = field(default_factory=dict)

    def export(self) -> Dict[str, object]:
        """转换为 JSON 兼容字典，整数键改为字符串键。"""
        return {
            "shared_pool_mode": self.mode,
            "max_shared_experts": self.max_shared_experts,
            "shared_probe_steps": self.probe_steps,
            "shared_similarity_threshold": self.similarity_threshold,
            "shared_expert_count": self.shared_expert_count,
            "task_to_shared": {str(k): int(v) for k, v in self.task_to_shared.items()},
            "shared_histories": {
                str(k): [int(tid) for tid in tids]
                for k, tids in self.shared_histories.items()
            },
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "SharedPoolState":
        """从新旧配置恢复共享池；旧 checkpoint 缺字段时使用安全默认值。"""
        mode = str(config.get("shared_pool_mode", "single"))
        default_shared_count = 0 if mode == "clustered" else 1
        task_to_shared = {
            int(k): int(v)
            for k, v in dict(config.get("task_to_shared", {})).items()
        }
        shared_histories = {
            int(k): [int(tid) for tid in tids]
            for k, tids in dict(config.get("shared_histories", {})).items()
        }
        return cls(
            mode=mode,
            max_shared_experts=int(config.get("max_shared_experts", 3)),
            probe_steps=int(config.get("shared_probe_steps", 100)),
            similarity_threshold=float(config.get("shared_similarity_threshold", 0.02)),
            shared_expert_count=int(config.get("shared_expert_count", default_shared_count)),
            task_to_shared=task_to_shared,
            shared_histories=shared_histories,
        )


def normalize_shared_pool_state(state: SharedPoolState) -> SharedPoolState:
    """复制并规范化共享池状态，保证计数、映射和历史记录一致。"""
    state = copy.deepcopy(state)
    if state.mode not in {"single", "clustered"}:
        raise ValueError(f"shared_pool_mode must be single or clustered, got {state.mode}")
    state.max_shared_experts = max(1, int(state.max_shared_experts))
    state.probe_steps = max(0, int(state.probe_steps))
    min_count = 0 if state.mode == "clustered" else 1
    state.shared_expert_count = max(min_count, int(state.shared_expert_count))
    state.shared_expert_count = min(state.shared_expert_count, state.max_shared_experts)
    state.task_to_shared = {int(k): int(v) for k, v in state.task_to_shared.items()}
    state.shared_histories = {
        int(k): [int(tid) for tid in tids]
        for k, tids in state.shared_histories.items()
    }
    if not state.shared_histories:
        state.shared_histories = {sid: [] for sid in range(state.shared_expert_count)}
    else:
        for sid in range(state.shared_expert_count):
            state.shared_histories.setdefault(sid, [])
    return state


def _as_float_vector(tensor: torch.Tensor) -> torch.Tensor:
    """将任意设备/精度的参数展平为 CPU FP32 向量，便于稳定比较。"""
    return tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)


def _safe_cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> Optional[float]:
    """计算余弦相似度，并把零向量、NaN、Inf 统一降级为零。"""
    lhs = _as_float_vector(lhs)
    rhs = _as_float_vector(rhs)
    lhs_norm = torch.linalg.vector_norm(lhs)
    rhs_norm = torch.linalg.vector_norm(rhs)
    if lhs_norm.item() == 0.0 or rhs_norm.item() == 0.0:
        return 0.0
    value = torch.dot(lhs, rhs) / (lhs_norm * rhs_norm)
    if torch.isnan(value) or torch.isinf(value):
        return 0.0
    return float(value.item())


def lora_delta(pair: LoraPair) -> torch.Tensor:
    """将 LoRA 的 A、B 矩阵还原为等价的低秩权重增量 B @ A。"""
    lora_a, lora_b = pair
    return lora_b.detach().to(torch.float32) @ lora_a.detach().to(torch.float32)


def compute_lora_similarity(lhs: LoraDeltaMap, rhs: LoraDeltaMap) -> float:
    """对两组同名线性层的 LoRA 增量余弦相似度求平均。"""
    scores: List[float] = []
    for module_name in sorted(set(lhs).intersection(rhs)):
        score = _safe_cosine(lora_delta(lhs[module_name]), lora_delta(rhs[module_name]))
        if score is not None:
            scores.append(score)
    if not scores:
        return 0.0
    value = sum(scores) / len(scores)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def lora_delta_map_from_state_dict(
    state_dict: Mapping[str, torch.Tensor], lora_name: str
) -> LoraDeltaMap:
    """从 state_dict 抽取指定 LoRA 名称的所有 A/B 参数对。"""
    escaped = re.escape(lora_name)
    a_pattern = re.compile(rf"^(.*\.loras_\.{escaped})\.lora_a_\.weight$")
    deltas: LoraDeltaMap = {}
    for key, lora_a in state_dict.items():
        match = a_pattern.match(key)
        if not match:
            continue
        prefix = match.group(1)
        lora_b_key = f"{prefix}.lora_b_.weight"
        if lora_b_key not in state_dict:
            continue
        module_name = prefix[: -len(f".loras_.{lora_name}")]
        deltas[module_name] = (lora_a, state_dict[lora_b_key])
    return deltas


def decide_shared_assignment(
    similarities: Mapping[int, float],
    current_count: int,
    max_shared_experts: int,
    threshold: float,
) -> Tuple[int, str, float]:
    """根据最大相似度和池容量，返回 CREATE 或 REUSE 的共享专家决策。"""
    if current_count <= 0:
        return 0, "CREATE", 0.0

    if similarities:
        selected_id, max_similarity = max(
            similarities.items(), key=lambda item: (item[1], -item[0])
        )
    else:
        selected_id, max_similarity = 0, 0.0

    if max_similarity < threshold and current_count < max_shared_experts:
        return current_count, "CREATE", max_similarity
    return int(selected_id), "REUSE", float(max_similarity)


def consolidate_parameter_pair(
    stored_param: torch.Tensor,
    working_param: torch.Tensor,
    history_size_before: int,
) -> float:
    """按 1/(历史任务数+1) 对单个共享参数做在线平均。"""
    weight = 1.0 / float(history_size_before + 1)
    with torch.no_grad():
        stored_param.add_(weight * (working_param.to(stored_param.device) - stored_param))
    return weight


def consolidate_state_dicts(
    stored: Mapping[str, torch.Tensor],
    working: Mapping[str, torch.Tensor],
    history_size_before: int,
    names: Optional[Iterable[str]] = None,
) -> float:
    """对一组同名参数执行在线平均，用于合并临时共享专家。"""
    selected_names = list(names) if names is not None else sorted(set(stored).intersection(working))
    weight = 1.0 / float(history_size_before + 1)
    with torch.no_grad():
        for name in selected_names:
            if name not in stored or name not in working:
                raise KeyError(f"Missing parameter during consolidation: {name}")
            stored[name].add_(weight * (working[name].to(stored[name].device) - stored[name]))
    return weight
