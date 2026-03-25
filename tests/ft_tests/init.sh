#! /bin/bash

git clone https://github.com/Limerances/Megatron-LM-FT.git
git checkout ft
cp -r Megatron-LM-FT/tests/ft_tests/* ./

mkdir checkpoint
mkdir data
mkdir nccl_trace

pip install transformers
pip install nvidia-resiliency-ext

bash preprocess.sh

bash run.sh |& tee log.txt