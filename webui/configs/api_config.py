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

# FastAPI服务器配置 (FastAPI Server Configuration)
FASTAPI_SERVER_CONFIG = {
    'debug': True,                 # 是否启用调试模式 (Enable debug mode)
    'title': 'SmallDoge API',        # API标题 (API title)
    'description': 'SmallDoge API for inference and training',  # API描述 (API description)
    'version': '0.0.1',           # API版本 (API version)
    'openapi_url': '/openapi.json',  # OpenAPI规范URL (OpenAPI spec URL)
    'docs_url': '/docs',             # Swagger文档URL (Swagger docs URL)
    'redoc_url': '/redoc',           # ReDoc文档URL (ReDoc docs URL)
    'openapi_prefix': '',            # OpenAPI前缀 (OpenAPI prefix)
}


# CORS配置 (CORS Configuration)
CORS_CONFIG = {
    'cors_origins': ['*'],           # 允许的跨域来源 (Allowed cross-origin sources)
    'allow_methods': ['*'],           # 允许的HTTP方法 (Allowed HTTP methods)
    'allow_headers': ['*'],           # 允许的HTTP头部 (Allowed HTTP headers)
    'allow_credentials': True,        # 允许携带凭证 (Allow credentials)
}


# Uvicorn服务器配置 (Uvicorn Server Configuration)
UNICORN_CONFIG = {
    'host': '127.0.0.1', # API服务器主机地址 (API server host address)
    'port': 8000, # API服务器端口 (API server port)
    'workers': 4, # 工作进程数量 (Number of worker processes)
    'reload': True, # 代码更改时自动重载 (Auto reload on code changes)
}