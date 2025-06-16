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
Model management models for SmallDoge WebUI
"""

import time
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, String, Integer, JSON, Boolean, Text

from smalldoge_webui.internal.db import Base
from smalldoge_webui.constants import MODEL_TYPES


####################
# Database Models
####################

class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    
    # Model configuration
    base_model_id = Column(String, nullable=True)
    model_type = Column(String, default=MODEL_TYPES.TRANSFORMERS)
    
    # Model parameters and metadata
    params = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)
    
    # Model access control
    access_control = Column(JSON, nullable=True)
    
    # Model status
    is_active = Column(Boolean, default=True)
    is_loaded = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()))


####################
# Pydantic Models
####################

class ModelParams(BaseModel):
    """Model parameters for inference"""
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    do_sample: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    num_beams: Optional[int] = 1
    early_stopping: Optional[bool] = False
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None


class ModelMeta(BaseModel):
    """Model metadata"""
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    capabilities: Optional[List[str]] = None
    context_length: Optional[int] = None
    model_size: Optional[str] = None
    license: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    version: Optional[str] = None
    architecture: Optional[str] = None
    training_data: Optional[str] = None
    languages: Optional[List[str]] = None


class ModelAccessControl(BaseModel):
    """Model access control settings"""
    read: Optional[List[str]] = None  # User IDs with read access
    write: Optional[List[str]] = None  # User IDs with write access
    public: Optional[bool] = False  # Public access
    groups: Optional[List[str]] = None  # Group access


class ModelModel(BaseModel):
    """Model data model"""
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    base_model_id: Optional[str] = None
    model_type: str
    params: Optional[ModelParams] = None
    meta: Optional[ModelMeta] = None
    access_control: Optional[ModelAccessControl] = None
    is_active: bool
    is_loaded: bool
    created_at: int
    updated_at: int


####################
# Response Models
####################

class ModelResponse(ModelModel):
    """Model response"""
    pass


class ModelListResponse(BaseModel):
    """Response for model list endpoints"""
    models: List[ModelResponse]
    total: int


class OpenAIModelResponse(BaseModel):
    """OpenAI-compatible model response"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "smalldoge-webui"


class OpenAIModelsResponse(BaseModel):
    """OpenAI-compatible models list response"""
    object: str = "list"
    data: List[OpenAIModelResponse]


####################
# Form Models
####################

class ModelForm(BaseModel):
    """Form for creating/updating models"""
    id: str
    name: str
    base_model_id: Optional[str] = None
    model_type: str = MODEL_TYPES.TRANSFORMERS
    params: Optional[ModelParams] = None
    meta: Optional[ModelMeta] = None
    access_control: Optional[ModelAccessControl] = None
    is_active: bool = True


class ModelUpdateForm(BaseModel):
    """Form for updating model information"""
    name: Optional[str] = None
    params: Optional[ModelParams] = None
    meta: Optional[ModelMeta] = None
    access_control: Optional[ModelAccessControl] = None
    is_active: Optional[bool] = None


class ModelLoadForm(BaseModel):
    """Form for loading/unloading models"""
    model_id: str
    load: bool = True


####################
# Model Status Models
####################

class ModelStatus(BaseModel):
    """Model status information"""
    id: str
    name: str
    is_loaded: bool
    is_available: bool
    memory_usage: Optional[float] = None
    load_time: Optional[float] = None
    last_used: Optional[int] = None
    error: Optional[str] = None


class ModelHealth(BaseModel):
    """Model health check response"""
    model_id: str
    status: str  # healthy, unhealthy, loading, error
    message: Optional[str] = None
    timestamp: int


####################
# Model Configuration Models
####################

class ModelConfig(BaseModel):
    """Model configuration for loading"""
    model_name_or_path: str
    trust_remote_code: bool = True
    torch_dtype: Optional[str] = "auto"
    device_map: Optional[str] = "auto"
    low_cpu_mem_usage: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    attn_implementation: Optional[str] = None


class TokenizerConfig(BaseModel):
    """Tokenizer configuration"""
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = True
    use_fast: bool = True
    padding_side: str = "left"
    truncation_side: str = "right"
    add_special_tokens: bool = True


####################
# Utility Functions
####################

def get_default_model_params() -> ModelParams:
    """Get default model parameters"""
    return ModelParams(
        max_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_beams=1,
        early_stopping=False
    )


def get_default_model_meta() -> ModelMeta:
    """Get default model metadata"""
    return ModelMeta(
        description="A language model for chat completion",
        tags=["chat", "completion"],
        capabilities=["text-generation", "conversation"],
        context_length=2048,
        languages=["en"]
    )


def validate_model_id(model_id: str) -> bool:
    """Validate model ID format"""
    # Allow alphanumeric, hyphens, underscores, and forward slashes
    import re
    pattern = r'^[a-zA-Z0-9\-_/]+$'
    return bool(re.match(pattern, model_id))


def create_model_id(name: str) -> str:
    """Create a model ID from name"""
    import re
    # Replace spaces and special characters with hyphens
    model_id = re.sub(r'[^a-zA-Z0-9\-_/]', '-', name)
    # Remove multiple consecutive hyphens
    model_id = re.sub(r'-+', '-', model_id)
    # Remove leading/trailing hyphens
    model_id = model_id.strip('-')
    return model_id.lower()
