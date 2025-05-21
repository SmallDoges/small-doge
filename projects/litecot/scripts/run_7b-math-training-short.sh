#!/bin/bash

# 设置执行权限后运行: chmod +x run_training.sh

# 只使用后面4张卡（4-7）
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 使用accelerate启动训练
accelerate launch \
    --config_file /data/zhangjiayi/long2short/zero3.yaml \
    sft.py \
    --config /data/zhangjiayi/long2short/config7b-qwen2.5-math-short25k.yaml \
    2>&1 | tee training_7b_qwen2.5_math_short25k_log.txt