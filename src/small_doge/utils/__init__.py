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

from .model_utils import get_modeling_classes, register_model_classes
from .callbacks import DogeWarmupCallback, Doge2WarmupCallback
from .training_args_configs import PTConfig, SFTConfig, DPOConfig, GRPOConfig, GRPOScriptArguments
from .rewards import (
    accuracy_reward,
    format_reward,
    format_deepseek_reward,
    reasoning_steps_reward,
    len_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    REWARD_FUNCS_REGISTRY,
)

__all__ = [
    "get_modeling_classes",
    "register_model_classes", 
    "DogeWarmupCallback",
    "Doge2WarmupCallback",
    "PTConfig",
    "SFTConfig", 
    "DPOConfig",
    "GRPOConfig",
    "GRPOScriptArguments",
    "accuracy_reward",
    "format_reward",
    "format_deepseek_reward",
    "reasoning_steps_reward",
    "len_reward",
    "get_cosine_scaled_reward",
    "get_repetition_penalty_reward",
    "REWARD_FUNCS_REGISTRY",
]
