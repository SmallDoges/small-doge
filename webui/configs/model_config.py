# coding=utf-8
# Copyright 2025 SmallDoge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# 模型配置 (Model Configuration)
# 此配置用于定义模型加载参数
# This configuration is used for defining model loading parameters
MODEL_CONFIG = {
    'model_name_or_path': 'Qwen/Qwen2.5-0.5B-Instruct',  # 模型名称或路径 (Model name or path)
    'device_map': 'auto',  # 设备映射 (Device mapping)
    'dtype': 'bfloat16',  # 数据类型 (Data type)
}


# 模型生成配置 (Model Generation Configuration)
# 此配置用于定义模型生成参数
# This configuration is used for defining model loading and generation parameters
MODEL_GENERATION_CONFIG = {
    'max_new_tokens': 1024,  # 最大生成token数 (Maximum number of tokens to generate)
    'temperature': 0.7,  # 采样温度，控制随机性 (Sampling temperature, controls randomness)
    'top_p': 0.9,  # 核采样参数，控制词汇多样性 (Top-p nucleus sampling parameter, controls vocabulary diversity)
    'top_k': 50,  # 最高采样参数，控制词汇多样性 (Top-k sampling parameter, controls vocabulary diversity)
    "repetition_penalty": 1.0,  # 重复惩罚参数，控制重复生成 (Repetition penalty parameter, controls repeated generation)
    'use_cache': True,  # 是否使用缓存 (Whether to use cache)
}