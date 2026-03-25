#!/bin/bash

VOCAB_FILE=gpt2_vocab/vocab.json
MERGE_FILE=gpt2_vocab/merges.txt
RAW_DATA_PATH=raw_data/shakespeare.jsonl
DATA_PATH=data/my_shakespeare
CHECKPOINT_PATH=checkpoint/megatron_moe_test

export HF_ENDPOINT=https://hf-mirror.com

python Megatron-LM-FT/tools/preprocess_data.py \
       --input $RAW_DATA_PATH \
       --output-prefix $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --tokenizer-type GPT2BPETokenizer \
       --json-keys text \
       --append-eod \
       --workers 1