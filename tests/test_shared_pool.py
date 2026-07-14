import math
import unittest

import torch

from mlora.common.modelargs import LLMModelArgs, MixConfig
from mlora.common.mix_lora import MixtralSparseMoe
from mlora.common.shared_pool import (
    SharedPoolState,
    compute_lora_similarity,
    consolidate_parameter_pair,
    decide_shared_assignment,
)


def make_delta(seed: int, zero: bool = False):
    torch.manual_seed(seed)
    a = torch.randn(2, 4)
    b = torch.randn(3, 2)
    if zero:
        b.zero_()
    return {"layer.mlp.w1": (a, b)}


class SharedPoolTest(unittest.TestCase):
    def test_similarity_handles_identical_random_and_zero(self):
        lhs = make_delta(7)
        identical = {
            name: (pair[0].clone(), pair[1].clone())
            for name, pair in lhs.items()
        }
        random_other = make_delta(9)
        zero = make_delta(11, zero=True)

        identical_score = compute_lora_similarity(lhs, identical)
        random_score = compute_lora_similarity(lhs, random_other)
        zero_score = compute_lora_similarity(lhs, zero)

        self.assertGreater(identical_score, 0.999)
        self.assertLess(random_score, identical_score)
        self.assertFalse(math.isnan(zero_score))

    def test_assignment_decisions(self):
        self.assertEqual(decide_shared_assignment({}, 0, 3, 0.02)[:2], (0, "CREATE"))
        self.assertEqual(decide_shared_assignment({0: 0.0}, 1, 3, 0.02)[:2], (1, "CREATE"))
        self.assertEqual(decide_shared_assignment({0: 0.5}, 1, 3, 0.02)[:2], (0, "REUSE"))
        self.assertEqual(decide_shared_assignment({0: -0.1, 1: 0.0, 2: -0.2}, 3, 3, 0.02)[:2], (1, "REUSE"))

    def test_consolidation_formula(self):
        stored = torch.tensor([1.0, 3.0])
        working = torch.tensor([3.0, 7.0])
        weight = consolidate_parameter_pair(stored, working, history_size_before=1)
        self.assertAlmostEqual(weight, 0.5)
        self.assertTrue(torch.allclose(stored, torch.tensor([2.0, 5.0])))

    def test_gate_stays_two_way(self):
        args = LLMModelArgs(dim_=4, hidden_act_="silu", device_=torch.device("cpu"))
        config = MixConfig().from_config({
            "name": "moe-cl",
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.1,
            "target_modules": {"w1_proj": True, "w2_proj": True, "w3_proj": True},
            "num_experts": 5,
            "shared_pool_mode": "clustered",
            "max_shared_experts": 3,
            "shared_probe_steps": 1,
            "shared_similarity_threshold": 0.02,
        }).check()
        config.adapter_name = "moe-cl"
        moe = MixtralSparseMoe(args, config, layer_ind=0)
        self.assertEqual(moe.gate_.out_features, 2)

    def test_mock_three_task_histories_keep_task_specific_ids(self):
        state = SharedPoolState(mode="clustered", max_shared_experts=2, shared_expert_count=0)
        task_specific_ids = []
        for task_id, similarities in [(1, {}), (2, {0: 0.0}), (3, {0: 0.9, 1: 0.1})]:
            selected, decision, _ = decide_shared_assignment(
                similarities,
                state.shared_expert_count,
                state.max_shared_experts,
                threshold=0.02,
            )
            if decision == "CREATE":
                state.shared_expert_count = selected + 1
                state.shared_histories.setdefault(selected, [])
            state.task_to_shared[task_id] = selected
            state.shared_histories.setdefault(selected, []).append(task_id)
            task_specific_ids.append(f"expert{task_id}")

        self.assertEqual(state.task_to_shared, {1: 0, 2: 1, 3: 0})
        self.assertEqual(state.shared_histories, {0: [1, 3], 1: [2]})
        self.assertEqual(task_specific_ids, ["expert1", "expert2", "expert3"])


if __name__ == "__main__":
    unittest.main()
