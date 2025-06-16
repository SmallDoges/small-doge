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
Constants for SmallDoge WebUI
"""

# Error Messages
class ERROR_MESSAGES:
    DEFAULT = lambda err="": f"Something went wrong {'(' + err + ')' if err else ''}."
    
    # Removed authentication errors - open access WebUI
    
    # Model Errors
    MODEL_NOT_FOUND = lambda name="": f"Model '{name}' not found."
    MODEL_LOAD_ERROR = lambda name="", err="": f"Failed to load model '{name}'{': ' + err if err else ''}."
    MODEL_INFERENCE_ERROR = lambda err="": f"Model inference failed{': ' + err if err else ''}."
    
    # Chat Errors
    CHAT_NOT_FOUND = "Chat not found."
    MESSAGE_NOT_FOUND = "Message not found."
    INVALID_CHAT_FORMAT = "Invalid chat format."
    
    # File Errors
    FILE_NOT_FOUND = "File not found."
    FILE_TOO_LARGE = lambda size="": f"File too large{': ' + size if size else ''}."
    INVALID_FILE_FORMAT = "Invalid file format."
    
    # API Errors
    INVALID_REQUEST = "Invalid request."
    RATE_LIMIT_EXCEEDED = "Rate limit exceeded."
    SERVICE_UNAVAILABLE = "Service temporarily unavailable."

# Success Messages
class SUCCESS_MESSAGES:
    DEFAULT = "Operation completed successfully."
    CHAT_CREATED = "Chat created successfully."
    CHAT_UPDATED = "Chat updated successfully."
    CHAT_DELETED = "Chat deleted successfully."
    MODEL_LOADED = "Model loaded successfully."

# Model Types
class MODEL_TYPES:
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    OLLAMA = "ollama"

# Chat Message Roles
class MESSAGE_ROLES:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

# API Endpoints
class ENDPOINTS:
    # Chats
    CHATS = "/api/v1/chats"
    CHAT_BY_ID = "/api/v1/chats/{chat_id}"

    # Models
    MODELS = "/api/v1/models"
    MODEL_BY_ID = "/api/v1/models/{model_id}"

    # OpenAI Compatible
    OPENAI_MODELS = "/openai/models"
    OPENAI_CHAT_COMPLETIONS = "/openai/chat/completions"

# HTTP Status Codes
class STATUS_CODES:
    OK = 200
    CREATED = 201
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    INTERNAL_SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503

# Default Configuration Values
class DEFAULTS:
    # Model Parameters
    MAX_TOKENS = 2048
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    
    # Chat Parameters
    CHAT_HISTORY_LIMIT = 100
    MESSAGE_LENGTH_LIMIT = 10000
    
    # File Upload
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES = [".txt", ".md", ".pdf", ".docx"]
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 3600  # 1 hour

# Model Configuration
class MODEL_CONFIG:
    # SmallDoge Models - synchronized with frontend
    SMALLDOGE_MODELS = [
        "SmallDoge/Doge-160M",
        "SmallDoge/Doge-60M",
        "SmallDoge/Doge-160M-Instruct",
        "SmallDoge/Doge-320M",
    ]
    
    # Supported HuggingFace Tasks
    SUPPORTED_TASKS = [
        "text-generation",
        "conversational", 
        "question-answering",
        "text-classification",
        "summarization",
        "translation",
        "text2text-generation"
    ]
    
    # Model Categories for UI
    MODEL_CATEGORIES = {
        "featured": "Featured Models",
        "text-generation": "Text Generation",
        "conversational": "Chat & Conversation", 
        "question-answering": "Question Answering",
        "code": "Code Generation",
        "instruct": "Instruction Following",
        "small": "Small & Efficient",
        "multilingual": "Multilingual"
    }
    
    # Default Model Parameters
    DEFAULT_PARAMS = {
        "max_tokens": DEFAULTS.MAX_TOKENS,
        "temperature": DEFAULTS.TEMPERATURE,
        "top_p": DEFAULTS.TOP_P,
        "top_k": DEFAULTS.TOP_K,
        "do_sample": True,
        "pad_token_id": None,  # Will be set based on model
        "eos_token_id": None,  # Will be set based on model
    }
    
    # Model Loading Parameters
    LOAD_PARAMS = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

# Database Configuration
class DB_CONFIG:
    # Table Names
    CHATS_TABLE = "chats"
    MESSAGES_TABLE = "messages"
    MODELS_TABLE = "models"

    # Connection Pool
    POOL_SIZE = 10
    MAX_OVERFLOW = 20
    POOL_TIMEOUT = 30
    POOL_RECYCLE = 3600

# Logging Configuration
class LOGGING:
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Log Levels
    LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }
