#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
# 本地单卡运行配置
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# 显存不够时的大杀器：强制清空缓存
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

VOCAB_FILE=gpt2_vocab/vocab.json
MERGE_FILE=gpt2_vocab/merges.txt
DATA_PATH=data/my_shakespeare_text_document
CHECKPOINT_PATH=checkpoint/megatron_moe_test

TENSORBOARD=tensorboard

WANDB_PROJECT="megatron-moe-test"
WANDB_EXP_NAME="moe-32exp-1gpu-run1"

#在collect文件夹下创建时间戳文件夹 中国时间
COLLECT_BASE_PATH=collect
TIMESTAMP=$(TZ=UTC-8 date +"%Y%m%d_%H%M%S")
export COLLECT_PATH=$COLLECT_BASE_PATH/$TIMESTAMP
mkdir -p $COLLECT_PATH


torchrun --nproc_per_node 1 Megatron-LM-FT/pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 16 \
    --global-batch-size 16 \
    --train-iters 1000 \
    --lr 0.00015 \
    --bf16 \
    --use-flash-attn \
    --recompute-activations \
    --num-experts 32 \
    --moe-router-topk 2 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-path $DATA_PATH \
    --save-interval 1001 \
    --split 949,50,1 \
    --eval-interval 1001 \
    --eval-iters 1 \
    --swiglu \
    --normalization RMSNorm \
    --disable-bias-linear \
    --log-timers-to-tensorboard \
    --tensorboard-dir $TENSORBOARD \
    --moe-per-layer-logging \
    --moe-aux-loss-coeff 0.01 \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH \
    # --wandb-project $WANDB_PROJECT \
    # --wandb-exp-name $WANDB_EXP_NAME \