SECONDS=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
BASE_MODEL="${BASE_MODEL:-../model/Llama-2-7b-hf}"

python mlora.py \
    --base_model "${BASE_MODEL}" \
    --config configs/mtl5/moe-cl-clustered.json \
    --train_task dbpedia \
    --order order1_clustered

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

elapsed_time=$SECONDS
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

echo "Elapsed time: $hours hours, $minutes minutes, and $seconds seconds"
