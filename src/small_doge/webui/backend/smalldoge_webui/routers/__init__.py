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
API routers for SmallDoge WebUI
Open source feature sharing - no authentication required
"""

from .chats import router as chats_router
from .models import router as models_router
from .openai import router as openai_router
from .huggingface import router as huggingface_router
from .datasets import router as datasets_router
from .training import router as training_router

__all__ = [
    "chats_router",
    "models_router",
    "openai_router",
    "huggingface_router",
    "datasets_router",
    "training_router",
]
