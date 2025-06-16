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
Utility functions for SmallDoge WebUI
Open source feature sharing - no authentication required
"""

from .models import (
    load_model,
    unload_model,
    get_loaded_models,
    get_available_models,
    is_model_loaded,
)

from .chat import (
    generate_chat_completion,
    format_chat_messages,
    create_chat_response,
)

__all__ = [
    # Model utilities
    "load_model",
    "unload_model",
    "get_loaded_models",
    "get_available_models",
    "is_model_loaded",

    # Chat utilities
    "generate_chat_completion",
    "format_chat_messages",
    "create_chat_response",
]

