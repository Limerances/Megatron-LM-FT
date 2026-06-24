#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Megatron-LM-FT

: "${CUDA_VISIBLE_DEVICES:=1,2,3,4,5}"
export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING=${TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING:-false}
if [[ -x /usr/local/cuda-13.1/bin/ptxas ]]; then
    export PATH=/usr/local/cuda-13.1/bin:${PATH}
    export TRITON_PTXAS_PATH=/usr/local/cuda-13.1/bin/ptxas
fi
export PYTHONPATH=/workspace/racer:${PYTHONPATH:-}

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-6000}
TRAIN_ITERS=${TRAIN_ITERS:-100}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
EVAL_ITERS=${EVAL_ITERS:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/workspace/checkpoints/gpt2_1.5b}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-/workspace/logs/gpt2_1.5b/tensorboard}

if [[ "${RACER_ENABLE:-1}" != "1" ]]; then
    mkdir -p "${CHECKPOINT_DIR}"
fi
mkdir -p "${TENSORBOARD_DIR}"

RACER_FLAGS=()
if [[ "${RACER_ENABLE:-1}" == "1" ]]; then
    RACER_FLAGS=(
        --racer-checkpoint
        --racer-path /workspace/racer
        --racer-k "${RACER_K:-3}"
        --racer-m "${RACER_M:-1}"
        --racer-train-ranks "${RACER_TRAIN_RANKS:-0,1,2,3}"
        --racer-spare-ranks "${RACER_SPARE_RANKS:-4}"
        --racer-buffer-size "${RACER_BUFFER_SIZE:-67108864}"
        --racer-retain-checkpoints "${RACER_RETAIN_CHECKPOINTS:-1}"
    )

    if [[ "${RACER_ASYNC_OFFLOAD:-0}" == "1" ]]; then
        RACER_FLAGS+=(--racer-async-offload)
    fi

    if [[ "${RACER_DISTRIBUTED_STORE:-1}" == "1" ]]; then
        RACER_FLAGS+=(--racer-distributed-store)
    fi

    if [[ "${RACER_VERIFY_ON_SAVE:-0}" == "1" ]]; then
        RACER_FLAGS+=(--racer-verify-on-save --racer-verify-rank "${RACER_VERIFY_RANK:-0}")
    fi

    if [[ "${RACER_VERIFY_LOAD_AFTER_SAVE:-0}" == "1" ]]; then
        RACER_FLAGS+=(--racer-verify-load-checkpoint-after-save)
    fi

    if [[ -n "${RACER_RECOVER_RANKS:-}" ]]; then
        RACER_FLAGS+=(--racer-recover-ranks "${RACER_RECOVER_RANKS}")
    fi

    if [[ "${RACER_FORCE_RECOVER:-0}" == "1" ]]; then
        RACER_FLAGS+=(--racer-force-recover)
    fi
fi

# RACER memory adapter keeps legacy torch state_dicts in GPU memory; do not add --use-dist-ckpt here.
torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port "${MASTER_PORT}" \
    pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --num-layers 48 \
    --hidden-size 1600 \
    --ffn-hidden-size 6400 \
    --num-attention-heads 25 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --attention-backend auto \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --train-iters "${TRAIN_ITERS}" \
    --lr 1.5e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters "${LR_WARMUP_ITERS:-10}" \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --bf16 \
    --use-distributed-optimizer \
    --ckpt-format torch \
    --data-path /workspace/data/my_shakespeare_text_document \
    --vocab-file /workspace/gpt2_vocab/vocab.json \
    --merge-file /workspace/gpt2_vocab/merges.txt \
    --split 949,50,1 \
    --save "${CHECKPOINT_DIR}" \
    --load "${CHECKPOINT_DIR}" \
    --tensorboard-dir "${TENSORBOARD_DIR}" \
    --log-interval "${LOG_INTERVAL:-10}" \
    --save-interval "${SAVE_INTERVAL}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-iters "${EVAL_ITERS}" \
    "${RACER_FLAGS[@]}"
