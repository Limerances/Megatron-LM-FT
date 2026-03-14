#! /bin/bash

git clone https://github.com/Limerances/Megatron-LM-FT.git
git checkout ft
cp -r Megatron-LM-FT/tests/ft_tests/* ./
bash preprocess.sh

