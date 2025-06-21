# Copyright 2025 The SmallDoge Team. All rights reserved.
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

"""
SmallDoge Dataset Processors

This module provides dataset processing utilities for different training tasks:
- Pre-training (PT): Text data processing with tokenization and packing
- Supervised Fine-tuning (SFT): Conversation data with chat templates
- Direct Preference Optimization (DPO): Preference data with prompt/chosen/rejected
- Group Relative Preference Optimization (GRPO): Math reasoning data with reward functions
"""

from .pt_datasets_process import mix_datasets_by_ratio as mix_pt_datasets
from .sft_datasets_process import mix_datasets_by_ratio as mix_sft_datasets  
from .dpo_datasets_process import mix_datasets_by_ratio as mix_dpo_datasets
from .grpo_datasets_process import mix_datasets_by_ratio as mix_grpo_datasets

__all__ = [
    "mix_pt_datasets",
    "mix_sft_datasets", 
    "mix_dpo_datasets",
    "mix_grpo_datasets",
]
