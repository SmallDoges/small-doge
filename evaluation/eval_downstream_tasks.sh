#!/bin/bash

MODEL="SmallDoge/Doge-160M"
MODEL_ARGS="pretrained=$MODEL,trust_remote_code=True,dtype=bfloat16,max_length=2048,generation_parameters={max_new_tokens:2048,temperature:0.6,top_p:0.95}"
OUTPUT_DIR="./lighteval_results/$MODEL"

if [[ $MODEL =~ Instruct$ ]]; then
    lighteval accelerate $MODEL_ARGS \
    "evaluation/instruct/doge_instruct.txt" \
    --custom-tasks evaluation/instruct/tasks.py \
    --override-batch-size 1 \
    --output-dir $OUTPUT_DIR
elif [[ $MODEL =~ Reason$ ]]; then
    lighteval vllm $MODEL_ARGS \
    "evaluation/reason/doge_reason.txt" \
    --custom-tasks evaluation/reason/tasks.py \
    --override-batch-size 1 \
    --output-dir $OUTPUT_DIR
else
    lighteval accelerate $MODEL_ARGS \
    "evaluation/base/doge_base.txt" \
    --custom-tasks evaluation/base/tasks.py \
    --override-batch-size 1 \
    --output-dir $OUTPUT_DIR
fi
