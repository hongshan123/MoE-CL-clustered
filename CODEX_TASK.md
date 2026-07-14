你正在修改当前 MoE-CL 官方代码仓库。

请先完整阅读 README、训练入口、模型构建代码、LoRA expert 实现、mix_lora/feed_forward 相关代码、checkpoint 保存加载逻辑和 MTL5 训练脚本，然后直接修改代码并运行必要的静态检查和轻量测试。

====================
一、研究目标
====================

原始 MoE-CL 使用：

1 个 global shared LoRA expert
+
每个 task 一个 task-specific LoRA expert

当前任务 t 已知 task_id。

原始模型对当前任务使用：

shared LoRA S
+
task-specific LoRA L_t

并通过原有 2-way gate：

[beta_s, beta_t] = G(z_i)

融合：

z_{i+1} = beta_s * z_s + beta_t * z_t

不要修改 task-specific LoRA 的设计。
不要合并 task-specific LoRA。
不要删除 task_id。
不要实现 task-agnostic routing。
不要把原来的 2-way gate 改成 K-way gate。

本次修改只针对 global shared LoRA。

目标是：

将单个 global shared LoRA：

S

改造成一个有容量上限 K 的 shared LoRA expert pool：

S = {S_0, S_1, ..., S_{M-1}}, M <= K

不同但相似的任务共享同一个 shared LoRA expert。

核心思想：

global universal sharing
->
similarity-aware clustered sharing

方法参考 K-Merge 的 LoRA parameter-space cosine similarity，
但应用对象是 MoE-CL 的 shared LoRA，而不是 task-specific LoRA。

====================
二、必须保留原始 MoE-CL 模式
====================

必须实现两种运行模式：

1. single
   完全复现原始 MoE-CL，一个 global shared LoRA。

2. clustered
   使用新的 shared LoRA pool。

增加配置项，命名可以根据当前代码风格调整，但语义必须明确：

shared_pool_mode: single | clustered
max_shared_experts: int
shared_probe_steps: int
shared_similarity_threshold: float

默认：

shared_pool_mode = single

保证原始代码、原始配置、原始 checkpoint 尽可能兼容。

不要为了新方法大规模重构整个项目。

====================
三、Clustered Shared LoRA 的训练流程
====================

对于顺序到达的新任务 T_t，执行以下阶段。

--------------------
Stage A: Probe Training
--------------------

为当前任务创建一个临时 probe LoRA：

P_t

P_t 必须是独立 LoRA，不加入正式 shared pool，也不是 task-specific LoRA L_t。

冻结：

- pretrained backbone
- 已有 shared LoRA experts
- 历史 task-specific LoRA experts

仅使用当前任务训练数据，对 P_t 进行短训练。

训练步数由：

shared_probe_steps

控制。

Probe 的目标只是获得当前任务的 LoRA parameter update direction。

Probe 结束后：

对于 P_t 中的每个 LoRA target module，得到：

Delta W_t = B_t @ A_t

不要只比较 A 或 B。
必须比较完整 LoRA update matrix B @ A。

--------------------
Stage B: Shared Expert Similarity
--------------------

对于已有 shared expert S_k：

计算每一个对应 LoRA module 的：

Delta W_k = B_k @ A_k

flatten Delta W_t 和 Delta W_k。

计算 cosine similarity。

对于所有匹配的 LoRA modules 求平均：

sim(P_t, S_k)
=
mean_m cosine(
    flatten(B_t^m @ A_t^m),
    flatten(B_k^m @ A_k^m)
)

其中 m 表示当前代码实际使用的 LoRA target module。

不要硬编码 q/k/v/o projection。
先检查 MoE-CL 实际 LoRA 注入位置，
根据真实 LoRA module 自动匹配 module name。

必须正确处理：

- module name 不匹配
- tensor device
- dtype
- zero norm
- NaN

提供独立、可测试的 similarity utility。

得到：

c = argmax_k sim(P_t, S_k)

--------------------
Stage C: Shared Expert Assignment
--------------------

实现如下决策：

如果 shared pool 为空：

    创建 S_0
    当前任务分配给 S_0

否则：

    找最大相似度：
    max_sim = max_k sim(P_t, S_k)

    如果：
        max_sim < shared_similarity_threshold
        AND
        current_shared_expert_count < max_shared_experts

    则：
        创建一个新的 shared expert
        新 expert 使用 probe LoRA P_t 的参数进行初始化

    否则：
        选择最相似 shared expert S_c

当 pool 已达到 max_shared_experts 后：

    禁止继续创建 shared expert

    必须：
        c = argmax similarity

维护：

task_id -> shared_expert_id

映射。

同时维护每个 shared expert 的 history：

shared_expert_id -> assigned task ids

例如：

shared 0 -> [0, 3, 7]
shared 1 -> [1, 5]
shared 2 -> [2, 4, 6]

该状态必须能够 checkpoint save/load。

--------------------
Stage D: MoE-CL Main Training
--------------------

确定 shared expert S_c 后：

复制 S_c：

working_shared_t = deepcopy(S_c)

正式训练当前任务时使用：

working_shared_t
+
task-specific LoRA L_t

原始 MoE-CL 的 shared/specific 结构必须保持。

计算：

z_s = F_LoRA(z_i, working_shared_t)
z_t = F_LoRA(z_i, L_t)

原有 gate 保持为 2-way gate：

[beta_s, beta_t] = G(z_i)

输出仍为：

z_{i+1}
=
beta_s * z_s
+
beta_t * z_t

不要让 gate 在多个 shared experts 中选择。
shared expert 在 task level 已经由 Stage C 选定。

训练当前任务时：

Train:
- working_shared_t
- current task-specific LoRA L_t
- 原始 MoE-CL 当前需要训练的 gate 参数
- 原始 discriminator / GAN 相关参数

Freeze:
- pretrained backbone
- shared pool 中正式保存的 S_0 ... S_{M-1}
- 历史 task-specific LoRAs

尽可能保持原始 MoE-CL：

L = L_SFT - alpha * L_GAN

的训练逻辑。

不要删除 GAN。
不要删除 task-aware discriminator。
不要改变原始 gate 输出维度 2。

--------------------
Stage E: Shared Delta Consolidation
--------------------

当前任务训练完成后计算：

Delta S_t
=
working_shared_t - S_c

这里是对应 LoRA 参数的 parameter delta。

不要直接执行：

S_c = working_shared_t

也不要直接平均 task-specific LoRA。

采用 history-aware consolidation。

设：

n_c = 当前任务加入前 S_c 已经承担的历史任务数量

对于已有 shared expert：

lambda_t = 1 / (n_c + 1)

更新：

S_c
=
S_c
+
lambda_t * Delta S_t

等价地对 shared LoRA 的所有 trainable parameters 执行：

param_Sc += lambda_t * (
    param_working_shared - param_Sc
)

如果这是新创建的 shared expert：

直接使用当前 main-training 完成后的 working_shared_t
作为该 shared expert 的最终参数。

然后：

将 task_id 添加到 shared expert history。

删除临时：

P_t
working_shared_t

释放不再需要的 GPU memory。

====================
四、GAN / Discriminator 处理
====================

原始 MoE-CL 是 global shared representation 的 task-aware adversarial learning。

新的 clustered 模式中，先保持原始 GAN loss 实现和训练机制，
不要擅自重新设计新的 loss。

但是代码结构必须支持当前选中的 shared_expert_id。

确保 discriminator 使用的是：

当前 working_shared_t 的 shared representation

而不是：

所有 shared experts 的输出

也不是：

task-specific representation。

在日志中记录：

task_id
shared_expert_id
cluster history

为后续实现 cluster-local discriminator 做准备。

当前版本先不增加新的 cluster-level GAN objective。

====================
五、Checkpoint 与状态保存
====================

clustered 模式 checkpoint 必须保存：

1. 所有 shared LoRA experts 参数
2. task-specific LoRA 参数
3. task_to_shared mapping
4. shared expert histories
5. shared expert count
6. clustered shared 配置
7. 原有 MoE-CL checkpoint 状态

建议增加一个明确的 shared pool state。

例如：

{
  "task_to_shared": {
    "0": 0,
    "1": 1,
    "2": 0
  },
  "shared_histories": {
    "0": [0, 2],
    "1": [1]
  }
}

具体格式按照现有项目 checkpoint 架构实现。

旧 MoE-CL checkpoint 如果只有一个 shared expert：

在 clustered 模式加载时允许：

old shared expert -> shared expert 0

并初始化：

shared expert count = 1

尽量提供 backward compatibility。

====================
六、Logging
====================

对于每一个新任务，打印或记录：

[Shared Probe]
task_id
probe_steps

[Shared Similarity]
shared_0: similarity
shared_1: similarity
...
max_similarity

[Shared Assignment]
task_id
selected_shared_id
decision = CREATE or REUSE
pool_size
max_shared_experts

[Shared Consolidation]
shared_id
history_size_before
lambda
history_size_after

日志必须便于后续论文实验分析。

====================
七、配置和运行脚本
====================

在现有 MTL5 配置基础上增加一个 clustered shared 实验配置。

不要覆盖原始配置。

增加新的运行脚本，例如：

scripts/mtl5/run_clustered_shared_moe-cl.sh

或按照项目现有命名规范命名。

至少能够配置：

shared_pool_mode=clustered
max_shared_experts=3
shared_probe_steps=100
shared_similarity_threshold=0.02

具体配置格式必须遵守当前仓库真实配置系统。

不要自己创建一个与项目无关的新 argparse 系统。

====================
八、测试
====================

至少完成以下检查。

1. Python syntax / import check。

2. similarity utility test：
   - identical LoRA delta similarity 接近 1
   - orthogonal/random LoRA similarity 更低
   - zero norm 不产生 NaN

3. assignment test：
   - empty pool -> CREATE shared 0
   - low similarity and pool not full -> CREATE
   - high similarity -> REUSE nearest
   - pool full -> REUSE nearest regardless threshold

4. consolidation test：
   验证：

   S_new =
   S_old + 1/(n+1) * (S_working - S_old)

5. gate compatibility：
   确认 gate 输�[200~cd ~/MoE-CL

git checkout -b clustered-shared-lora

cat > CODEX_TASK.md <<'EOF'
你正在修改当前 MoE-CL 官方代码仓库。

请先完整阅读 README、训练入口、模型构建代码、LoRA expert 实现、mix_lora/feed_forward 相关代码、checkpoint 保存加载逻辑和 MTL5 训练脚本，然后直接修改代码并运行必要的静态检查和轻量测试。

====================
一、研究目标
====================

原始 MoE-CL 使用：

1 个 global shared LoRA expert
+
每个 task 一个 task-specific LoRA expert

当前任务 t 已知 task_id。

原始模型对当前任务使用：

shared LoRA S
+
task-specific LoRA L_t

并通过原有 2-way gate：

[beta_s, beta_t] = G(z_i)

融合：

z_{i+1} = beta_s * z_s + beta_t * z_t

不要修改 task-specific LoRA 的设计。
不要合并 task-specific LoRA。
不要删除 task_id。
不要实现 task-agnostic routing。
不要把原来的 2-way gate 改成 K-way gate。

本次修改只针对 global shared LoRA。

目标是：

将单个 global shared LoRA：

S

改造成一个有容量上限 K 的 shared LoRA expert pool：

S = {S_0, S_1, ..., S_{M-1}}, M <= K

不同但相似的任务共享同一个 shared LoRA expert。

核心思想：

global universal sharing
->
similarity-aware clustered sharing

方法参考 K-Merge 的 LoRA parameter-space cosine similarity，
但应用对象是 MoE-CL 的 shared LoRA，而不是 task-specific LoRA。

====================
二、必须保留原始 MoE-CL 模式
====================

必须实现两种运行模式：

1. single
   完全复现原始 MoE-CL，一个 global shared LoRA。

2. clustered
   使用新的 shared LoRA pool。

增加配置项，命名可以根据当前代码风格调整，但语义必须明确：

shared_pool_mode: single | clustered
max_shared_experts: int
shared_probe_steps: int
shared_similarity_threshold: float

默认：

shared_pool_mode = single

保证原始代码、原始配置、原始 checkpoint 尽可能兼容。

不要为了新方法大规模重构整个项目。

====================
三、Clustered Shared LoRA 的训练流程
====================

对于顺序到达的新任务 T_t，执行以下阶段。

--------------------
Stage A: Probe Training
--------------------

为当前任务创建一个临时 probe LoRA：

P_t

P_t 必须是独立 LoRA，不加入正式 shared pool，也不是 task-specific LoRA L_t。

冻结：

- pretrained backbone
- 已有 shared LoRA experts
- 历史 task-specific LoRA experts

仅使用当前任务训练数据，对 P_t 进行短训练。

训练步数由：

shared_probe_steps

控制。

Probe 的目标只是获得当前任务的 LoRA parameter update direction。

Probe 结束后：

对于 P_t 中的每个 LoRA target module，得到：

Delta W_t = B_t @ A_t

不要只比较 A 或 B。
必须比较完整 LoRA update matrix B @ A。

--------------------
Stage B: Shared Expert Similarity
--------------------

对于已有 shared expert S_k：

计算每一个对应 LoRA module 的：

Delta W_k = B_k @ A_k

flatten Delta W_t 和 Delta W_k。

计算 cosine similarity。

对于所有匹配的 LoRA modules 求平均：

sim(P_t, S_k)
=
mean_m cosine(
    flatten(B_t^m @ A_t^m),
    flatten(B_k^m @ A_k^m)
)

其中 m 表示当前代码实际使用的 LoRA target module。

不要硬编码 q/k/v/o projection。
先检查 MoE-CL 实际 LoRA 注入位置，
根据真实 LoRA module 自动匹配 module name。

必须正确处理：

- module name 不匹配
- tensor device
- dtype
- zero norm
- NaN

提供独立、可测试的 similarity utility。

得到：

c = argmax_k sim(P_t, S_k)

--------------------
Stage C: Shared Expert Assignment
--------------------

实现如下决策：

如果 shared pool 为空：

    创建 S_0
    当前任务分配给 S_0

否则：

    找最大相似度：
    max_sim = max_k sim(P_t, S_k)

    如果：
        max_sim < shared_similarity_threshold
        AND
        current_shared_expert_count < max_shared_experts

    则：
        创建一个新的 shared expert
        新 expert 使用 probe LoRA P_t 的参数进行初始化

    否则：
        选择最相似 shared expert S_c

当 pool 已达到 max_shared_experts 后：

    禁止继续创建 shared expert

    必须：
        c = argmax similarity

维护：

task_id -> shared_expert_id

映射。

同时维护每个 shared expert 的 history：

shared_expert_id -> assigned task ids

例如：

shared 0 -> [0, 3, 7]
shared 1 -> [1, 5]
shared 2 -> [2, 4, 6]

该状态必须能够 checkpoint save/load。

--------------------
Stage D: MoE-CL Main Training
--------------------

确定 shared expert S_c 后：

复制 S_c：

working_shared_t = deepcopy(S_c)

正式训练当前任务时使用：

working_shared_t
+
task-specific LoRA L_t

原始 MoE-CL 的 shared/specific 结构必须保持。

计算：

z_s = F_LoRA(z_i, working_shared_t)
z_t = F_LoRA(z_i, L_t)

原有 gate 保持为 2-way gate：

[beta_s, beta_t] = G(z_i)

输出仍为：

z_{i+1}
=
beta_s * z_s
+
beta_t * z_t

不要让 gate 在多个 shared experts 中选择。
shared expert 在 task level 已经由 Stage C 选定。

训练当前任务时：

Train:
- working_shared_t
- current task-specific LoRA L_t
- 原始 MoE-CL 当前需要训练的 gate 参数
- 原始 discriminator / GAN 相关参数

Freeze:
- pretrained backbone
- shared pool 中正式保存的 S_0 ... S_{M-1}
- 历史 task-specific LoRAs

尽可能保持原始 MoE-CL：

L = L_SFT - alpha * L_GAN

的训练逻辑。

不要删除 GAN。
不要删除 task-aware discriminator。
不要改变原始 gate 输出维度 2。

--------------------
Stage E: Shared Delta Consolidation
--------------------

当前任务训练完成后计算：

Delta S_t
=
working_shared_t - S_c

这里是对应 LoRA 参数的 parameter delta。

不要直接执行：

S_c = working_shared_t

也不要直接平均 task-specific LoRA。

采用 history-aware consolidation。

设：

n_c = 当前任务加入前 S_c 已经承担的历史任务数量

对于已有 shared expert：

lambda_t = 1 / (n_c + 1)

更新：

S_c
=
S_c
+
lambda_t * Delta S_t

等价地对 shared LoRA 的所有 trainable parameters 执行：

param_Sc += lambda_t * (
    param_working_shared - param_Sc
)

如果这是新创建的 shared expert：

直接使用当前 main-training 完成后的 working_shared_t
作为该 shared expert 的最终参数。

然后：

将 task_id 添加到 shared expert history。

删除临时：

P_t
working_shared_t

释放不再需要的 GPU memory。

====================
四、GAN / Discriminator 处理
====================

原始 MoE-CL 是 global shared representation 的 task-aware adversarial learning。

新的 clustered 模式中，先保持原始 GAN loss 实现和训练机制，
不要擅自重新设计新的 loss。

但是代码结构必须支持当前选中的 shared_expert_id。

确保 discriminator 使用的是：

当前 working_shared_t 的 shared representation

而不是：

所有 shared experts 的输出

也不是：

task-specific representation。

在日志中记录：

task_id
shared_expert_id
cluster history

为后续实现 cluster-local discriminator 做准备。

当前版本先不增加新的 cluster-level GAN objective。

====================
五、Checkpoint 与状态保存
====================

clustered 模式 checkpoint 必须保存：

1. 所有 shared LoRA experts 参数
2. task-specific LoRA 参数
3. task_to_shared mapping
4. shared expert histories
5. shared expert count
6. clustered shared 配置
7. 原有 MoE-CL checkpoint 状态

建议增加一个明确的 shared pool state。

例如：

{
  "task_to_shared": {
    "0": 0,
    "1": 1,
    "2": 0
  },
  "shared_histories": {
    "0": [0, 2],
    "1": [1]
  }
}

具体格式按照现有项目 checkpoint 架构实现。

旧 MoE-CL checkpoint 如果只有一个 shared expert：

在 clustered 模式加载时允许：

old shared expert -> shared expert 0

并初始化：

shared expert count = 1

尽量提供 backward compatibility。

====================
六、Logging
====================

对于每一个新任务，打印或记录：

[Shared Probe]
task_id
probe_steps

[Shared Similarity]
shared_0: similarity
shared_1: similarity
...
max_similarity

[Shared Assignment]
task_id
selected_shared_id
decision = CREATE or REUSE
pool_size
max_shared_experts

[Shared Consolidation]
shared_id
history_size_before
lambda
history_size_after

日志必须便于后续论文实验分析。

====================
七、配置和运行脚本
====================

在现有 MTL5 配置基础上增加一个 clustered shared 实验配置。

不要覆盖原始配置。

增加新的运行脚本，例如：

scripts/mtl5/run_clustered_shared_moe-cl.sh

或按照项目现有命名规范命名。

至少能够配置：

shared_pool_mode=clustered
max_shared_experts=3
shared_probe_steps=100
shared_similarity_threshold=0.02

具体配置格式必须遵守当前仓库真实配置系统。

不要自己创建一个与项目无关的新 argparse 系统。

====================
八、测试
====================

至少完成以下检查。

1. Python syntax / import check。

2. similarity utility test：
   - identical LoRA delta similarity 接近 1
   - orthogonal/random LoRA similarity 更低
   - zero norm 不产生 NaN

3. assignment test：
   - empty pool -> CREATE shared 0
   - low similarity and pool not full -> CREATE
   - high similarity -> REUSE nearest
   - pool full -> REUSE nearest regardless threshold

4. consolidation test：
   验证：

   S_new =
   S_old + 1/(n+1) * (S_working - S_old)

5. gate compatibility：
   确认 gate 输出仍然为最后一维 2。

6. smoke test：
   使用项目中能够轻量运行的方式，
   模拟至少 3 个 sequential tasks，
   验证 task_to_shared/history 正确更新。

如果完整 Llama 模型或数据不可用，
构造最小 mock/unit test，
不要因为模型权重不存在而放弃测试。

====================
九、禁止事项
====================

再次强调：

- 不要对 task-specific LoRA 做 K-Merge
- 不要限制 task-specific LoRA 数量
- 不要合并 task-specific LoRA
- 不要删除 task_id
- 不要实现 no-task-ID routing
- 不要实现 K-way shared gate
- 不要修改原始 2-way shared/specific gate 语义
- 不要删除 GAN
- 不要把所有 shared experts 同时前向
- 不要直接照搬 K-Merge 的完整 adapter management pipeline
- 不要大规模重写 MoE-CL

我们只做：

single global shared LoRA
->
bounded similarity-aware shared LoRA pool

流程为：

Probe Training
->
LoRA Delta Similarity
->
Shared Expert Assignment
->
Shared-Specific Joint Training
->
History-Aware Shared Delta Consolidation

====================
十、最终输出
====================

完成修改后：

1. 列出修改和新增的文件。
2. 对每个文件说明修改目的。
3. 给出核心训练流程。
4. 给出新的 clustered shared 训练命令。
5. 给出运行原始 MoE-CL baseline 的命令。
6. 给出测试结果。
7. 明确说明还有哪些 TODO 或风险。
8. 不要只给方案，必须直接修改仓库代码。
