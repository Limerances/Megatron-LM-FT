#!/bin/bash
set -euo pipefail

# Simple end-to-end EGM verification script:
# train -> periodic EGM save -> hang -> ft_launcher restarts -> load from EGM -> continue
#
# This script is intended for your real environment only.
# It should NOT be executed locally without the required GPUs / ft_launcher runtime.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# =========================
# 你主要改下面这几项即可
# =========================
VOCAB_FILE=gpt2_vocab/vocab.json
MERGE_FILE=gpt2_vocab/merges.txt
DATA_PATH=data/my_shakespeare_text_document

LOG_DIR=/tmp/megatron_egm_verify
LOG_FILE="${LOG_DIR}/verify_egm_restart.log"

NPROC_PER_NODE=4
TRAIN_ITERS=80
EGM_SAVE_INTERVAL=10
EGM_POOL_SIZE_GB=4
EGM_NUM_SLOTS=2
EGM_NUMA_NODE_ID=0
EGM_SOCKET_PATH=/tmp/megatron_egm_manager.sock

# Simulate one hung rank so ft_launcher kills and restarts the job.
FT_SIM_FAULT_DESC="rank_hung;1;120"

mkdir -p "${LOG_DIR}"

export NPROC_PER_NODE
export EGM_POOL_SIZE_GB
export EGM_NUM_SLOTS
export EGM_NUMA_NODE_ID
export EGM_SOCKET_PATH
export EGM_DEVICE_ID=0
export EGM_DAEMON_NUM_SLOTS=$((NPROC_PER_NODE * EGM_NUM_SLOTS))
export FT_SIM_FAULT_DESC

echo "Log file: ${LOG_FILE}"
echo "FT_SIM_FAULT_DESC=${FT_SIM_FAULT_DESC}"
echo "EGM_SOCKET_PATH=${EGM_SOCKET_PATH}"

set +e
bash "${ROOT_DIR}/scripts/start_megatron_with_egm.sh" \
    ft_launcher \
    --max-restarts 1 \
    --ft-rank-section-timeout=setup:60,step:30,checkpointing:420 \
    --ft-rank-out-of-section-timeout 300 \
    --nproc_per_node "${NPROC_PER_NODE}" "${ROOT_DIR}/pretrain_gpt.py" \
    --enable-ft-package \
    --calc-ft-timeouts \
    --tensor-model-parallel-size "${NPROC_PER_NODE}" \
    --pipeline-model-parallel-size 1 \
    --num-layers 4 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 16 \
    --global-batch-size 16 \
    --train-iters "${TRAIN_ITERS}" \
    --lr 0.00015 \
    --bf16 \
    --use-flash-attn \
    --recompute-activations \
    --num-experts 32 \
    --moe-router-topk 2 \
    --vocab-file "${VOCAB_FILE}" \
    --merge-file "${MERGE_FILE}" \
    --data-path "${DATA_PATH}" \
    --save-interval 1001 \
    --split 949,50,1 \
    --eval-interval 1001 \
    --eval-iters 1 \
    --swiglu \
    --normalization RMSNorm \
    --disable-bias-linear \
    --moe-per-layer-logging \
    --moe-aux-loss-coeff 0.01 \
    --sequence-parallel \
    --moe-router-dtype fp32 \
    --enable-egm-checkpoint \
    --egm-use-daemon \
    --egm-pool-size-gb "${EGM_POOL_SIZE_GB}" \
    --egm-num-slots "${EGM_NUM_SLOTS}" \
    --egm-save-interval "${EGM_SAVE_INTERVAL}" \
    --egm-daemon-socket-path "${EGM_SOCKET_PATH}" \
    --egm-numa-node-id "${EGM_NUMA_NODE_ID}" \
    2>&1 | tee "${LOG_FILE}"
launcher_rc=${PIPESTATUS[0]}
set -e

echo "ft_launcher exit code: ${launcher_rc}"

save_count="$(grep -c "Saved checkpoint to slot" "${LOG_FILE}" || true)"
load_count="$(grep -c "Loaded checkpoint from EGM at iteration" "${LOG_FILE}" || true)"

echo "save_count=${save_count}"
echo "load_count=${load_count}"

if [[ "${save_count}" -lt 1 ]]; then
    echo "ERROR: Did not observe any EGM checkpoint save." >&2
    echo "Hint: increase TRAIN_ITERS or reduce EGM_SAVE_INTERVAL." >&2
    exit 10
fi

if [[ "${load_count}" -lt 1 ]]; then
    echo "ERROR: Did not observe loading from EGM after restart." >&2
    echo "Hint: check whether the fault was actually triggered and ft_launcher really restarted the training job." >&2
    exit 11
fi

echo "SUCCESS: verified EGM save + restart + restore flow."
