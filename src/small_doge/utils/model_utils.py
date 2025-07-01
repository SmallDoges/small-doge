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

import os
from typing import Tuple, Type
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


def get_modeling_classes(recipe_type: str) -> Tuple[Type, Type, Type]:
    """
    Determine the model classes to use based on the recipe type in config
    
    Args:
        recipe_type (str): The type of recipe, e.g., 'doge' or 'doge2'.
        
    Returns:
        tuple: (Config class, Model class, ForCausalLM class)
    """
    # Determine whether it's doge or doge2 by recipe_type
    if recipe_type == 'doge2':
        from small_doge.models.doge2.modeling_doge2 import Doge2Config, Doge2Model, Doge2ForCausalLM
        return Doge2Config, Doge2Model, Doge2ForCausalLM
    else:
        from small_doge.models.doge.modeling_doge import DogeConfig, DogeModel, DogeForCausalLM
        return DogeConfig, DogeModel, DogeForCausalLM


def register_model_classes(recipe_type: str) -> None:
    """
    Register the corresponding model classes based on the recipe type in config
    
    Args:
        recipe_type (str): The type of recipe, e.g., 'doge' or 'doge2'.
    """
    config_class, model_class, causal_lm_class = get_modeling_classes(recipe_type)
    # Register the classes with AutoConfig, AutoModel, and AutoModelForCausalLM
    if recipe_type == 'doge2':
        AutoConfig.register("doge2", config_class)
        AutoModel.register(config_class, model_class)
        AutoModelForCausalLM.register(config_class, causal_lm_class)
    else:
        AutoConfig.register("doge", config_class)
        AutoModel.register(config_class, model_class)
        AutoModelForCausalLM.register(config_class, causal_lm_class)
