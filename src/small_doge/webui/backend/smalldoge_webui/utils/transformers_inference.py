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
Transformers-based model inference for SmallDoge WebUI
Handles model loading, inference, and streaming responses
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, AsyncGenerator, List
from contextlib import asynccontextmanager

import torch
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModelForCausalLM, 
    TextIteratorStreamer,
    GenerationConfig
)
from threading import Thread

from small_doge.webui.backend.smalldoge_webui.env import (
    MODEL_CACHE_DIR, 
    DEVICE, 
    TORCH_DTYPE,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K
)
from small_doge.webui.backend.smalldoge_webui.constants import MODEL_CONFIG, ERROR_MESSAGES
from small_doge.webui.backend.smalldoge_webui.models.chats import ChatMessage, ChatCompletionRequest
from small_doge.webui.backend.smalldoge_webui.models.models import ModelParams

log = logging.getLogger(__name__)


####################
# Model Manager
####################

class ModelManager:
    """Manages loaded models and tokenizers"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.device = self._get_device()
        self.torch_dtype = self._get_torch_dtype()
    
    def _get_device(self) -> str:
        """Determine the best device for inference"""
        if DEVICE != "auto":
            return DEVICE
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Determine the best torch dtype"""
        if TORCH_DTYPE == "auto":
            if self.device == "cuda":
                return torch.float16
            else:
                return torch.float32
        elif TORCH_DTYPE == "float16":
            return torch.float16
        elif TORCH_DTYPE == "float32":
            return torch.float32
        else:
            return torch.float32
    
    async def load_model(self, model_id: str, **kwargs) -> bool:
        """Load a model and tokenizer"""
        try:
            if model_id in self.models:
                log.info(f"Model {model_id} already loaded")
                return True
            
            log.info(f"Loading model: {model_id}")
            start_time = time.time()
            
            # Prepare loading parameters
            load_params = MODEL_CONFIG.LOAD_PARAMS.copy()
            load_params.update(kwargs)
            
            # Set device and dtype
            if self.device != "auto":
                load_params["device_map"] = None
            load_params["torch_dtype"] = self.torch_dtype
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=load_params.get("trust_remote_code", True),
                use_fast=True
            )
            
            # Set padding token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=MODEL_CACHE_DIR,
                **load_params
            )
            
            # Move to device if not using device_map
            if load_params.get("device_map") is None:
                model = model.to(self.device)
            
            # Extract model configuration information
            model_config = AutoConfig.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR, trust_remote_code=load_params.get("trust_remote_code", True))
            config_info = {
                "model_type": getattr(model_config, "model_type", "unknown"),
                "vocab_size": getattr(model_config, "vocab_size", None),
                "hidden_size": getattr(model_config, "hidden_size", None),
                "num_attention_heads": getattr(model_config, "num_attention_heads", None),
                "num_hidden_layers": getattr(model_config, "num_hidden_layers", None),
                "max_position_embeddings": getattr(model_config, "max_position_embeddings", None),
                "context_length": getattr(model_config, "max_position_embeddings", None) or "unknown",
                "architectures": getattr(model_config, "architectures", []),
            }
            
            # Create a serializable copy of load_params
            serializable_params = {}
            for key, value in load_params.items():
                if hasattr(value, '__dict__') or hasattr(value, '__name__'):
                    # Convert objects to string representation
                    serializable_params[key] = str(value)
                elif isinstance(value, (torch.dtype, type)):
                    # Convert torch types to string
                    serializable_params[key] = str(value)
                else:
                    # Keep primitive types as-is
                    serializable_params[key] = value
            
            # Store model and tokenizer
            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer
            self.model_configs[model_id] = {
                "load_time": time.time() - start_time,
                "device": str(self.device),
                "torch_dtype": str(self.torch_dtype),
                "parameters": serializable_params,
                "config": config_info,
                "tokenizer_info": {
                    "vocab_size": tokenizer.vocab_size,
                    "model_max_length": getattr(tokenizer, "model_max_length", None),
                    "pad_token": str(tokenizer.pad_token) if tokenizer.pad_token else None,
                    "eos_token": str(tokenizer.eos_token) if tokenizer.eos_token else None,
                    "bos_token": str(getattr(tokenizer, "bos_token", None)) if getattr(tokenizer, "bos_token", None) else None,
                }
            }
            
            load_time = time.time() - start_time
            log.info(f"Model {model_id} loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            log.error(f"Failed to load model {model_id}: {e}")
            return False
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model and free memory"""
        try:
            if model_id not in self.models:
                log.warning(f"Model {model_id} not loaded")
                return True
            
            log.info(f"Unloading model: {model_id}")
            
            # Remove from memory
            del self.models[model_id]
            del self.tokenizers[model_id]
            del self.model_configs[model_id]
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            log.info(f"Model {model_id} unloaded successfully")
            return True
            
        except Exception as e:
            log.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded"""
        return model_id in self.models
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model IDs"""
        return list(self.models.keys())
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model"""
        if model_id not in self.model_configs:
            return None
        return self.model_configs[model_id].copy()


# Global model manager instance
model_manager = ModelManager()


####################
# Inference Functions
####################

def prepare_generation_config(params: Optional[ModelParams] = None) -> GenerationConfig:
    """Prepare generation configuration from parameters"""
    if params is None:
        params = ModelParams()
    
    config = GenerationConfig(
        max_new_tokens=params.max_tokens or MAX_TOKENS,
        temperature=params.temperature or TEMPERATURE,
        top_p=params.top_p or TOP_P,
        top_k=params.top_k or TOP_K,
        do_sample=params.do_sample if params.do_sample is not None else True,
        repetition_penalty=params.repetition_penalty or 1.0,
        length_penalty=params.length_penalty or 1.0,
        num_beams=params.num_beams or 1,
        early_stopping=params.early_stopping if params.early_stopping is not None else False,
        pad_token_id=params.pad_token_id,
        eos_token_id=params.eos_token_id,
        bos_token_id=params.bos_token_id,
    )
    
    return config


def format_chat_messages(messages: List[ChatMessage], tokenizer) -> str:
    """Format chat messages for the model"""
    # Check if tokenizer has a chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        try:
            # Convert to dict format for chat template
            message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
            formatted = tokenizer.apply_chat_template(
                message_dicts, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            log.warning(f"Failed to use chat template: {e}")
    
    # Fallback to simple formatting
    formatted_messages = []
    for message in messages:
        if message.role == "system":
            formatted_messages.append(f"System: {message.content}")
        elif message.role == "user":
            formatted_messages.append(f"User: {message.content}")
        elif message.role == "assistant":
            formatted_messages.append(f"Assistant: {message.content}")
    
    # Add prompt for assistant response
    formatted_messages.append("Assistant:")
    return "\n".join(formatted_messages)


async def generate_completion(
    model_id: str,
    messages: List[ChatMessage],
    params: Optional[ModelParams] = None,
    stream: bool = False
) -> AsyncGenerator[str, None]:
    """Generate completion for chat messages"""
    
    # Ensure model is loaded
    if not model_manager.is_model_loaded(model_id):
        if not await model_manager.load_model(model_id):
            raise Exception(ERROR_MESSAGES.MODEL_LOAD_ERROR(model_id))
    
    model = model_manager.models[model_id]
    tokenizer = model_manager.tokenizers[model_id]
    
    # Format messages
    prompt = format_chat_messages(messages, tokenizer)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Prepare generation config
    generation_config = prepare_generation_config(params)
    
    if stream:
        # Streaming generation
        streamer = TextIteratorStreamer(
            tokenizer, 
            timeout=60.0, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            "generation_config": generation_config,
            "streamer": streamer,
        }
        
        # Start generation in a separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they are generated
        for token in streamer:
            if token:
                yield token
        
        thread.join()
    else:
        # Non-streaming generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        yield generated_text


####################
# OpenAI-Compatible API Functions
####################

async def create_chat_completion_response(
    request: ChatCompletionRequest,
    stream: bool = False
) -> AsyncGenerator[Dict[str, Any], None]:
    """Create OpenAI-compatible chat completion response"""
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    if stream:
        # Streaming response
        async for token in generate_completion(
            request.model,
            request.messages,
            ModelParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ),
            stream=True
        ):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None
                }]
            }
            yield chunk
        
        # Final chunk
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield final_chunk
    else:
        # Non-streaming response
        content = ""
        async for token in generate_completion(
            request.model,
            request.messages,
            ModelParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ),
            stream=False
        ):
            content += token
        
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,  # TODO: Calculate actual token counts
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        yield response


####################
# Utility Functions
####################

async def load_model(model_id: str, **kwargs) -> bool:
    """Load a model"""
    return await model_manager.load_model(model_id, **kwargs)


async def unload_model(model_id: str) -> bool:
    """Unload a model"""
    return await model_manager.unload_model(model_id)


def is_model_loaded(model_id: str) -> bool:
    """Check if model is loaded"""
    return model_manager.is_model_loaded(model_id)


def get_loaded_models() -> List[str]:
    """Get loaded models"""
    return model_manager.get_loaded_models()


def get_available_models() -> List[str]:
    """Get available models"""
    return MODEL_CONFIG.SMALLDOGE_MODELS.copy()


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model information"""
    return model_manager.get_model_info(model_id)
