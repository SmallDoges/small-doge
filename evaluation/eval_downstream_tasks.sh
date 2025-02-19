#!/bin/bash

MODEL="SmallDoge/Doge-160M"
OUTPUT_DIR="./lighteval_results"

if [[ $MODEL =~ Instruct$ ]]; then
    lighteval accelerate "pretrained=$MODEL,max_length=2048,trust_remote_code=True" \
    "evaluation/instruct/doge_instruct.txt" \
    --custom-tasks evaluation/instruct/tasks.py \
    --override-batch-size 1 \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
elif [[ $MODEL =~ Reason$ ]]; then
    lighteval vllm "pretrained=$MODEL,max_model_length=32768,gpu_memory_utilisation=0.8,trust_remote_code=True" \
    "evaluation/reason/doge_reason.txt" \
    --custom-tasks evaluation/reason/tasks.py \
    --override-batch-size 1 \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
else
    lighteval accelerate "pretrained=$MODEL,max_length=2048,trust_remote_code=True" \
    "evaluation/base/doge_base.txt" \
    --custom-tasks evaluation/base/tasks.py \
    --override-batch-size 1 \
    --output-dir $OUTPUT_DIR
fi
