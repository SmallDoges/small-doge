#!/bin/bash

# 设置执行权限后运行: chmod +x run_training.sh

# 使用accelerate启动训练
accelerate launch \
    --config_file /data/zhangjiayi/long2short/zero3.yaml \
    sft.py \
    --config /data/zhangjiayi/long2short/config1.5b-qwen2.5-math-short25k.yaml \
    2>&1 | tee training_1.5b_qwen2.5_math_log_short25k.txt 