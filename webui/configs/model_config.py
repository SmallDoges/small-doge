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


import torch

# Model Configuration
# This configuration is used for defining model loading parameters
MODEL_CONFIG = {
    'model_name_or_path': 'Qwen/Qwen2.5-0.5B-Instruct',  # Model name or path
    'trust_remote_code': True,  # Whether to trust remote code
    'device_map': 'auto' if torch.cuda.is_available() else None,  # Device mapping
    'torch_dtype': torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,  # Data type
}


# Model Generation Configuration
# This configuration is used for defining model loading and generation parameters
MODEL_GENERATION_CONFIG = {
    'max_new_tokens': 1024,  # Maximum number of tokens to generate
    'temperature': 0.7,  # Sampling temperature, controls randomness
    'top_p': 0.9,  # Top-p nucleus sampling parameter, controls vocabulary diversity
    'top_k': 50,  # Top-k sampling parameter, controls vocabulary diversity
    "repetition_penalty": 1.0,  # Repetition penalty parameter, controls repeated generation
    'use_cache': True,  # Whether to use cache
}