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
OpenAI-compatible API router for SmallDoge WebUI
Provides OpenAI-compatible endpoints for chat completions and models
"""

import json
import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse

from small_doge.webui.backend.smalldoge_webui.models.chats import ChatCompletionRequest
from small_doge.webui.backend.smalldoge_webui.models.models import OpenAIModelResponse, OpenAIModelsResponse
# Authentication removed for open source sharing
from small_doge.webui.backend.smalldoge_webui.utils.chat import generate_chat_completion, format_chat_messages, validate_chat_messages
from small_doge.webui.backend.smalldoge_webui.utils.models import get_available_models, get_model_capabilities, get_model_context_length
from small_doge.webui.backend.smalldoge_webui.utils.task_manager import task_manager
from small_doge.webui.backend.smalldoge_webui.constants import ERROR_MESSAGES

log = logging.getLogger(__name__)
router = APIRouter()


####################
# Models Endpoint
####################

@router.get("/models", response_model=OpenAIModelsResponse)
async def list_models():
    """
    List available models (OpenAI-compatible)

    Returns:
        OpenAIModelsResponse: List of available models
    """
    try:
        available_models = get_available_models()
        
        model_list = []
        for model_id in available_models:
            model_response = OpenAIModelResponse(
                id=model_id,
                object="model",
                created=int(time.time()),
                owned_by="smalldoge-webui"
            )
            model_list.append(model_response)
        
        return OpenAIModelsResponse(
            object="list",
            data=model_list
        )
    
    except Exception as e:
        log.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/models/{model_id}", response_model=OpenAIModelResponse)
async def get_model(model_id: str):
    """
    Get specific model information (OpenAI-compatible)

    Args:
        model_id: Model identifier

    Returns:
        OpenAIModelResponse: Model information
    """
    try:
        available_models = get_available_models()
        
        if model_id not in available_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(model_id)
            )
        
        return OpenAIModelResponse(
            id=model_id,
            object="model",
            created=int(time.time()),
            owned_by="smalldoge-webui"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Chat Completions Endpoint
####################

@router.post("/chat/completions")
async def create_chat_completion(
    request: Request,
    chat_request: ChatCompletionRequest
):
    """
    Create chat completion (OpenAI-compatible) with enhanced streaming support

    Args:
        request: FastAPI request object
        chat_request: Chat completion request data

    Returns:
        StreamingResponse or Dict: Chat completion response
    """
    try:
        # Convert to dict for processing
        form_data = chat_request.model_dump()

        # Validate messages format
        if not validate_chat_messages(form_data["messages"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.INVALID_CHAT_FORMAT
            )

        # Check if model is available
        available_models = get_available_models()
        if form_data["model"] not in available_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data["model"])
            )

        # Format messages
        formatted_messages = format_chat_messages(form_data["messages"])
        form_data["messages"] = [msg.model_dump() for msg in formatted_messages]

        # Set default values with enhanced parameters
        form_data.setdefault("temperature", 0.7)
        form_data.setdefault("top_p", 0.9)
        form_data.setdefault("max_tokens", 2048)
        form_data.setdefault("stream", False)
        form_data.setdefault("stop", None)
        form_data.setdefault("presence_penalty", 0.0)
        form_data.setdefault("frequency_penalty", 0.0)

        # Enhanced streaming support
        if form_data.get("stream", False):
            response = await generate_chat_completion(
                request=request,
                form_data=form_data,
                user=None,  # No authentication required
                bypass_filter=False
            )

            # Ensure proper streaming headers
            if hasattr(response, 'headers'):
                response.headers["Cache-Control"] = "no-cache"
                response.headers["Connection"] = "keep-alive"
                response.headers["X-Accel-Buffering"] = "no"

            return response
        else:
            # Non-streaming response
            return await generate_chat_completion(
                request=request,
                form_data=form_data,
                user=None,  # No authentication required
                bypass_filter=False
            )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Chat completion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.MODEL_INFERENCE_ERROR(str(e))
        )


####################
# Completions Endpoint (Legacy)
####################

@router.post("/completions")
async def create_completion(
    request: Request,
    form_data: Dict[str, Any]
):
    """
    Create text completion (OpenAI-compatible, legacy)

    Args:
        request: FastAPI request object
        form_data: Completion request data

    Returns:
        Dict: Completion response
    """
    try:
        # Convert completion request to chat completion format
        if "prompt" not in form_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt is required"
            )
        
        if "model" not in form_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model is required"
            )
        
        # Convert prompt to chat messages
        chat_messages = [
            {"role": "user", "content": form_data["prompt"]}
        ]
        
        # Create chat completion request
        chat_form_data = {
            "model": form_data["model"],
            "messages": chat_messages,
            "temperature": form_data.get("temperature", 0.7),
            "top_p": form_data.get("top_p", 0.9),
            "max_tokens": form_data.get("max_tokens", 2048),
            "stream": form_data.get("stream", False),
            "stop": form_data.get("stop"),
            "presence_penalty": form_data.get("presence_penalty", 0.0),
            "frequency_penalty": form_data.get("frequency_penalty", 0.0),
            "user": form_data.get("user")
        }
        
        # Generate completion using chat completion endpoint
        response = await generate_chat_completion(
            request=request,
            form_data=chat_form_data,
            user=None,  # No authentication required
            bypass_filter=False
        )
        
        # Convert chat completion response to completion format
        if isinstance(response, StreamingResponse):
            return response
        else:
            # Convert response format
            if "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice:
                    choice["text"] = choice["message"]["content"]
                    del choice["message"]
            
            response["object"] = "text_completion"
            return response
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Completion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.MODEL_INFERENCE_ERROR(str(e))
        )


####################
# Task Management Endpoints
####################

@router.post("/chat/cancel/{task_id}")
async def cancel_chat_completion(task_id: str):
    """
    Cancel a running chat completion task
    
    Args:
        task_id: Task identifier to cancel
    
    Returns:
        Dict: Cancellation status
    """
    try:
        # Attempt to cancel the task
        cancelled = await task_manager.cancel_task(task_id)
        
        if cancelled:
            return {
                "status": "cancelled",
                "task_id": task_id,
                "message": "Task cancelled successfully"
            }
        else:
            # Task not found or not cancellable
            task = task_manager.get_task(task_id)
            if not task:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task {task_id} not found"
                )
            else:
                return {
                    "status": "not_cancelled",
                    "task_id": task_id,
                    "current_status": task.status.value,
                    "message": f"Task cannot be cancelled (current status: {task.status.value})"
                }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """
    Get the status of a task
    
    Args:
        task_id: Task identifier
    
    Returns:
        Dict: Task status information
    """
    try:
        task = task_manager.get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return {
            "task_id": task_id,
            "status": task.status.value,
            "model_id": task.model_id,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "error": task.error
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting task status {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/tasks/active")
async def get_active_tasks():
    """
    Get all active tasks
    
    Returns:
        Dict: List of active tasks
    """
    try:
        active_tasks = task_manager.get_active_tasks()
        
        return {
            "active_tasks": [
                {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "model_id": task.model_id,
                    "created_at": task.created_at,
                    "started_at": task.started_at
                }
                for task in active_tasks.values()
            ],
            "count": len(active_tasks)
        }
    
    except Exception as e:
        log.error(f"Error getting active tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/tasks/stats")
async def get_task_stats():
    """
    Get task statistics
    
    Returns:
        Dict: Task statistics
    """
    try:
        stats = task_manager.get_task_stats()
        
        return {
            "task_stats": stats,
            "total_tasks": sum(stats.values())
        }
    
    except Exception as e:
        log.error(f"Error getting task stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Model Information Endpoints
####################

@router.get("/models/{model_id}/capabilities")
async def get_model_capabilities_endpoint(model_id: str):
    """
    Get model capabilities

    Args:
        model_id: Model identifier

    Returns:
        Dict: Model capabilities
    """
    try:
        available_models = get_available_models()

        if model_id not in available_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(model_id)
            )

        capabilities = get_model_capabilities(model_id)
        context_length = get_model_context_length(model_id)

        return {
            "model_id": model_id,
            "capabilities": capabilities,
            "context_length": context_length,
            "supports_streaming": True,
            "supports_functions": False,
            "supports_vision": False
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting model capabilities for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Additional OpenAI-Compatible Endpoints
####################

@router.get("/engines")
async def list_engines():
    """
    List available engines (OpenAI legacy compatibility)

    Returns:
        Dict: List of available engines
    """
    try:
        available_models = get_available_models()

        engines = []
        for model_id in available_models:
            engine = {
                "id": model_id,
                "object": "engine",
                "owner": "smalldoge-webui",
                "ready": True
            }
            engines.append(engine)

        return {
            "object": "list",
            "data": engines
        }

    except Exception as e:
        log.error(f"Error listing engines: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/engines/{engine_id}")
async def get_engine(engine_id: str):
    """
    Get specific engine information (OpenAI legacy compatibility)

    Args:
        engine_id: Engine identifier

    Returns:
        Dict: Engine information
    """
    try:
        available_models = get_available_models()

        if engine_id not in available_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(engine_id)
            )

        return {
            "id": engine_id,
            "object": "engine",
            "owner": "smalldoge-webui",
            "ready": True
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting engine {engine_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.post("/embeddings")
async def create_embeddings(request: Request, form_data: Dict[str, Any]):
    """
    Create embeddings (placeholder for future implementation)

    Args:
        request: FastAPI request object
        form_data: Embeddings request data

    Returns:
        Dict: Embeddings response
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Embeddings endpoint not yet implemented. This is a placeholder for future functionality."
    )


@router.get("/usage")
async def get_usage():
    """
    Get API usage statistics

    Returns:
        Dict: Usage statistics
    """
    try:
        # Basic usage stats - can be enhanced with actual metrics
        return {
            "total_requests": 0,  # TODO: Implement actual tracking
            "total_tokens": 0,    # TODO: Implement actual tracking
            "models_used": get_available_models(),
            "uptime": "unknown",  # TODO: Implement uptime tracking
            "status": "operational"
        }

    except Exception as e:
        log.error(f"Error getting usage stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )
