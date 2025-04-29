# coding=utf-8
# Copyright 2025 Jingze Shi and the HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .models.doge.configuration_doge import DogeConfig
    from .models.doge.modeling_doge import DogeForCausalLM, DogeForSequenceClassification, DogeModel, DogePreTrainedModel
    from .models.doge2.configuration_doge2 import Doge2Config
    from .models.doge2.modeling_doge2 import Doge2ForCausalLM, Doge2ForSequenceClassification, Doge2Model, Doge2PreTrainedModel
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)

from .trainer.doge import (
    doge_pt_trainer,
    doge_sft_trainer,
    doge_dpo_trainer,
    doge_grpo_trainer,
    doge_tokenizer_trainer,
)
from .trainer.doge2 import (
    doge2_pt_trainer,
    doge2_sft_trainer,
    doge2_dpo_trainer,
    doge2_grpo_trainer,
    doge2_tokenizer_trainer,
)

__all__ = [
    "DogeConfig",
    "DogeForCausalLM",
    "DogeForSequenceClassification",
    "DogeModel",
    "DogePreTrainedModel",
    "Doge2Config",
    "Doge2ForCausalLM",
    "Doge2ForSequenceClassification",
    "Doge2Model",
    "Doge2PreTrainedModel",
    "doge_pt_trainer",
    "doge_sft_trainer",
    "doge_dpo_trainer",
    "doge_grpo_trainer",
    "doge_tokenizer_trainer",
    "doge2_pt_trainer",
    "doge2_sft_trainer",
    "doge2_dpo_trainer",
    "doge2_grpo_trainer",
    "doge2_tokenizer_trainer",
]
