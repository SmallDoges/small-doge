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

# SmallDoges WebUI Configuration
APP_CONFIG = {
    'debug': True,                 # Debug mode
    'host': '0.0.0.0',             # Server host address
    'port': 7860,                  # Server port
    'title': 'SmallDoges',         # Application title
    'description': 'All-in-one WebUI for Inference and Training',  # Application description
    'share': True,                 # Whether to share the application
    'auth_enabled': True,          # Whether to enable authentication
    'theme': 'default',            # UI theme
    'assets_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets'),  # Assets directory path
    'favicon': 'favicon.ico',      # Website icon
    'allowed_paths': ['assets'],   # Allowed access paths
    'quiet': True,                 # Quiet mode
    'show_api': False,             # Whether to show API
    'root_path': "",               # Root path
    'ssl_verify': False,           # SSL verification
    'prevent_thread_lock': True,   # Prevent thread lock
    'max_connections': 100,        # Maximum number of connections
    'timeout': 30,                 # Timeout duration in seconds
    'enable_logging': False        # Enable logging
}