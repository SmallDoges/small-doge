import os

# 模型配置 (Model Configuration)
# 此配置用于定义模型加载和生成参数
# This configuration is used for defining model loading and generation parameters

MODEL_GENERATION_CONFIG = {
    'model_name_or_path': os.environ.get('MODEL_NAME_OR_PATH', 'Qwen/Qwen2.5-0.5B-Instruct'),  # 模型名称或路径 (Model name or path)
    'max_new_tokens': 1024,  # 最大生成token数 (Maximum number of tokens to generate)
    'temperature': 0.7,  # 采样温度，控制随机性 (Sampling temperature, controls randomness)
    'top_p': 0.9,  # 核采样参数，控制词汇多样性 (Top-p nucleus sampling parameter, controls vocabulary diversity)
    'top_k': 50,  # 最高采样参数，控制词汇多样性 (Top-k sampling parameter, controls vocabulary diversity)
    "repetition_penalty": 1.0,  # 重复惩罚参数，控制重复生成 (Repetition penalty parameter, controls repeated generation)
    'use_cache': True,  # 是否使用缓存 (Whether to use cache)
    'system_prompt': "你是一个友好的助手，帮助用户回答问题。",  # 系统提示词 (System prompt)
}