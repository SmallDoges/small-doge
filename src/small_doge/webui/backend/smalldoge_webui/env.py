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
Environment variables configuration for SmallDoge WebUI
"""

import os
import logging
from pathlib import Path
from typing import Optional

# Base directories
SMALLDOGE_WEBUI_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", SMALLDOGE_WEBUI_DIR / "data")).resolve()
CACHE_DIR = Path(os.getenv("CACHE_DIR", DATA_DIR / "cache")).resolve()

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Environment
ENV = os.environ.get("ENV", "dev")

# Database
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR}/smalldoge_webui.db"
)

# Security
WEBUI_SECRET_KEY = os.environ.get(
    "WEBUI_SECRET_KEY", 
    "your-secret-key-change-this-in-production"
)

# Authentication (disabled for open source sharing)
WEBUI_AUTH = os.environ.get("WEBUI_AUTH", "False").lower() == "true"
ENABLE_SIGNUP = os.environ.get("ENABLE_SIGNUP", "False").lower() == "true"
DEFAULT_USER_ROLE = os.environ.get("DEFAULT_USER_ROLE", "user")

# JWT Configuration
JWT_EXPIRES_IN = os.environ.get("JWT_EXPIRES_IN", "7d")

# Model Configuration
DEFAULT_MODELS = os.environ.get("DEFAULT_MODELS", "SmallDoge/Doge-320M-Instruct")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", CACHE_DIR / "models")).resolve()
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Server Configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# CORS Configuration
ENABLE_CORS = os.environ.get("ENABLE_CORS", "True").lower() == "true"
CORS_ALLOW_ORIGIN = os.environ.get("CORS_ALLOW_ORIGIN", "*")

# Logging Configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
GLOBAL_LOG_LEVEL = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

# WebUI Configuration
WEBUI_NAME = os.environ.get("WEBUI_NAME", "SmallDoge WebUI")
WEBUI_FAVICON_URL = os.environ.get("WEBUI_FAVICON_URL", "/favicon.ico")

# Model Inference Configuration
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TOP_K = int(os.environ.get("TOP_K", "50"))

# Device Configuration
DEVICE = os.environ.get("DEVICE", "auto")  # auto, cpu, cuda, mps
TORCH_DTYPE = os.environ.get("TORCH_DTYPE", "auto")  # auto, float16, float32

# Safety and Content Filtering
ENABLE_CONTENT_FILTER = os.environ.get("ENABLE_CONTENT_FILTER", "False").lower() == "true"

# API Configuration
API_PREFIX = "/api/v1"
OPENAI_API_PREFIX = "/openai"

# Session Configuration
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "False").lower() == "true"
SESSION_COOKIE_SAME_SITE = os.environ.get("SESSION_COOKIE_SAME_SITE", "lax")

# Version and Build Info
VERSION = "0.1.0"
BUILD_HASH = os.environ.get("BUILD_HASH", "dev")

# Logging setup
logging.basicConfig(
    level=GLOBAL_LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

log = logging.getLogger(__name__)
log.info(f"SmallDoge WebUI Backend v{VERSION} starting...")
log.info(f"Environment: {ENV}")
log.info(f"Data directory: {DATA_DIR}")
log.info(f"Database URL: {DATABASE_URL}")
log.info(f"Model cache directory: {MODEL_CACHE_DIR}")
