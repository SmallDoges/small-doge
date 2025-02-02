
$MODEL = "JingzeShi/Doge-20M"
$OUTPUT_DIR = "./lighteval_results"

if ($MODEL -match "Instruct$") {
    lighteval accelerate "pretrained=$MODEL,max_length=2048,trust_remote_code=True" `
    "extended|ifeval|5|0" `
    --override-batch-size 1 `
    --output-dir $OUTPUT_DIR `
    --use-chat-template
} else {
    lighteval accelerate "pretrained=$MODEL,max_length=2048,trust_remote_code=True" `
    "original|mmlu|5|0,lighteval|triviaqa|5|0,lighteval|arc:easy|5|0,leaderboard|arc:challenge|5|0,lighteval|piqa|5|0,leaderboard|hellaswag|5|0,lighteval|openbookqa|5|0,leaderboard|winogrande|5|0" `
    --override-batch-size 1 `
    --output-dir $OUTPUT_DIR
}
