# coding=utf-8
# Copyright 2024 the SmallDoge team. All rights reserved.
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
Configuration classes for SmallDoge training.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import trl
from transformers import TrainingArguments
from trl import ScriptArguments


@dataclass
class PTConfig(TrainingArguments):
    """
    Configuration for Small-Doge Pre-Training.
    """
    recipe_type: str = field(
        default="doge",
        metadata={"help": "The type of recipe to use, e.g., 'doge'."},
    )

    model_init_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Additional keyword arguments for model initialization."}
    )

    # Dataset parameters
    datasets_and_ratios: Optional[List[Dict[str, float]]] = field(
        default=None,
        metadata={"help": "List of datasets and their mixing ratios. Format: [{'dataset_name': ratio}, ...]"}
    )
    total_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of samples to use from mixed datasets"}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "The field name containing text data in the dataset"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for tokenization"}
    )
    packing: bool = field(
        default=True,
        metadata={"help": "Whether to pack sequences for efficient training"}
    )
    dataset_num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes for dataset processing"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to cache processed datasets"}
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    Configuration for Small-Doge Supervised Fine-Tuning.
    """
    
    # Dataset mixing parameters
    datasets_and_ratios: Optional[List[Dict[str, float]]] = field(
        default=None,
        metadata={"help": "List of datasets and their mixing ratios. Format: [{'dataset_name': ratio}, ...]"}
    )
    total_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of samples to use from mixed datasets"}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "The field name containing text data in the dataset"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for tokenization"}
    )
    packing: bool = field(
        default=True,
        metadata={"help": "Whether to pack sequences for efficient training"}
    )
    dataset_num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes for dataset processing"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to cache processed datasets"}
    )
    tools: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={"help": "List of tools/functions for function calling support"}
    )


@dataclass
class DPOConfig(trl.DPOConfig):
    """
    Configuration for Small-Doge Direct Preference Optimization.
    """
    chat_template: Optional[str] = field(
        default=None, 
        metadata={"help": "The chat template to use."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    recipe_type: Optional[str] = field(
        default="doge",
        metadata={"help": "The type of recipe to use, e.g., 'doge' or 'doge2'."},
    )
    
    # Dataset mixing parameters
    datasets_and_ratios: Optional[List[Dict[str, float]]] = field(
        default=None,
        metadata={"help": "List of datasets and their mixing ratios. Format: [{'dataset_name': ratio}, ...]"}
    )
    total_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of samples to use from mixed datasets"}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "The field name containing text data in the dataset"}
    )
    tools: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={"help": "List of tools/functions for function calling support"}
    )


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    Configuration for Small-Doge Group Relative Preference Optimization.
    """
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    recipe_type: Optional[str] = field(
        default="doge",
        metadata={"help": "The type of recipe to use, e.g., 'doge' or 'doge2'."},
    )
    
    # Dataset mixing parameters
    datasets_and_ratios: Optional[List[Dict[str, float]]] = field(
        default=None,
        metadata={"help": "List of datasets and their mixing ratios. Format: [{'dataset_name': ratio}, ...]"}
    )
    total_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of samples to use from mixed datasets"}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "The field name containing text data in the dataset"}
    )
    dataset_num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes for dataset processing"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to cache processed datasets"}
    )
    tools: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={"help": "List of tools/functions for function calling support"}
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
