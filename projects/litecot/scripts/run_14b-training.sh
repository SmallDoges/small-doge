#!/bin/bash

# 设置执行权限后运行: chmod +x run_training.sh

# 使用accelerate启动训练
accelerate launch \
    --config_file /data/zhangjiayi/long2short/zero3_offload.yaml \
    sft.py \
    --config /data/zhangjiayi/long2short/config14b-qwen2.5.yaml \
    2>&1 | tee training_14b_qwen2.5_log.txt