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

from .pt import main as pt_trainer
from .sft import main as sft_trainer
from .dpo import main as dpo_trainer
from .grpo import main as grpo_trainer
from .train import main as unified_trainer

__all__ = [
    "pt_trainer",
    "sft_trainer", 
    "dpo_trainer",
    "grpo_trainer",
    "unified_trainer",
]
