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

# TODO: @wubingheng111 需要完善RAG配置

KNOWLEDGE_BASE_CONFIG = {
    'embedding_model': os.environ.get('EMBEDDING_MODEL', 'text-embedding-ada-002'),
    'top_k': int(os.environ.get('TOP_K', 5)),
    'similarity_threshold': float(os.environ.get('SIMILARITY_THRESHOLD', 0.75)),
    
}