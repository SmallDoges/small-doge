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
API_SERVER_CONFIG = {
    # 基本配置 (Basic Configuration)
    'host': '127.0.0.1',             # API服务器主机地址 (API server host address)
    'port': 8000,                    # API服务器端口 (API server port)
    'workers': 4,                    # 工作进程数量 (Number of worker processes)
    'reload': True,                  # 代码更改时自动重载 (Auto reload on code changes)
    'timeout': 120,                  # 请求超时时间(秒) (Request timeout in seconds)
    
    # CORS配置 (CORS Configuration)
    'cors_origins': ['*'],           # 允许的跨域来源 (Allowed cross-origin sources)
    'allow_methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],  # 允许的HTTP方法 (Allowed HTTP methods)
    'allow_headers': ['*'],          # 允许的HTTP头部 (Allowed HTTP headers)
    'allow_credentials': True,       # 允许携带凭证 (Allow credentials)
    
    # 安全配置 (Security Configuration)
    'ssl_enabled': True,             # 启用SSL (Enable SSL)
    'ssl_keyfile': os.path.join(os.path.dirname(__file__), 'certs', 'key.pem'),  # SSL密钥文件路径 (SSL key file path)
    'ssl_certfile': os.path.join(os.path.dirname(__file__), 'certs', 'cert.pem'),  # SSL证书文件路径 (SSL certificate file path)
    'auth_enabled': True,            # 启用API认证 (Enable API authentication)
    'api_key_header': 'X-API-Key',   # API密钥头部名称 (API key header name)
    
    # 性能配置 (Performance Configuration)
    'max_requests': 1000,            # 最大请求数 (Maximum requests)
    'max_connections': 100,          # 最大连接数 (Maximum connections)
    'backlog': 2048,                 # 挂起的连接队列大小 (Backlog size)
    'limit_concurrency': 500,        # 并发请求限制 (Concurrent request limit)
    'limit_request_line': 8190,      # 请求行字节限制 (Request line byte limit)
    
    # 日志配置 (Logging Configuration)
    'request_logging': True,         # 启用请求日志记录 (Enable request logging)
    'log_level': 'info',             # 日志级别 (Log level)
    'access_log': True,              # 访问日志 (Access log)
    'log_config': None,              # 自定义日志配置 (Custom logging configuration)
    
    # 文档配置 (Documentation Configuration)
    'docs_url': '/docs',             # Swagger文档URL (Swagger docs URL)
    'redoc_url': '/redoc',           # ReDoc文档URL (ReDoc docs URL)
    'openapi_url': '/openapi.json',  # OpenAPI规范URL (OpenAPI spec URL)
    'openapi_prefix': '',            # OpenAPI前缀 (OpenAPI prefix)
}