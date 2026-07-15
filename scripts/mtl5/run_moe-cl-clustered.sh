# 聚类共享 LoRA 的完整持续学习流程：DBpedia -> Amazon -> Yahoo -> AGNews。
SECONDS=0
# 可由调用者覆盖的设备、通信超时、显存分配策略与基础模型路径。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
BASE_MODEL="${BASE_MODEL:-../model/Llama-2-7b-hf}"

# 第一个任务从随机初始化的共享池开始训练。
python mlora.py \
    --base_model "${BASE_MODEL}" \
    --config configs/mtl5/moe-cl-clustered.json \
    --train_task dbpedia \
    --order order1_clustered

# 后续任务恢复上一任务 checkpoint，并更新共享专家池。
python mlora.py \
    --base_model "${BASE_MODEL}" \
    --config configs/mtl5/moe-cl-clustered.json \
    --train_task amazon \
    --load_adapter_file dbpedia_finetuned \
    --order order1_clustered

python mlora.py \
    --base_model "${BASE_MODEL}" \
    --config configs/mtl5/moe-cl-clustered.json \
    --train_task yahoo \
    --load_adapter_file amazon_finetuned \
    --order order1_clustered

python mlora.py \
    --base_model "${BASE_MODEL}" \
    --config configs/mtl5/moe-cl-clustered.json \
    --train_task agnews \
    --load_adapter_file yahoo_finetuned \
    --order order1_clustered

# 汇总整个四任务序列的耗时。
elapsed_time=$SECONDS
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

echo "Elapsed time: $hours hours, $minutes minutes, and $seconds seconds"
