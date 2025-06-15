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

# FastAPI Server Configuration
FASTAPI_SERVER_CONFIG = {
    'debug': True,                 # Enable debug mode
    'title': 'SmallDoge API',        # API title
    'description': 'SmallDoge API for inference and training',  # API description
    'version': '0.0.1',           # API version
    'openapi_url': '/openapi.json',  # OpenAPI spec URL
    'docs_url': '/docs',             # Swagger docs URL
    'redoc_url': '/redoc',           # ReDoc docs URL
    'openapi_prefix': '',            # OpenAPI prefix
}


# CORS Configuration
CORS_CONFIG = {
    'cors_origins': ['*'],           # Allowed cross-origin sources
    'allow_methods': ['*'],           # Allowed HTTP methods
    'allow_headers': ['*'],           # Allowed HTTP headers
    'allow_credentials': True,        # Allow credentials
}


# Uvicorn Server Configuration
UNICORN_CONFIG = {
    'host': '127.0.0.1', # API server host address
    'port': 8000, # API server port
    'workers': 4, # Number of worker processes
    'reload': True, # Auto reload on code changes
}