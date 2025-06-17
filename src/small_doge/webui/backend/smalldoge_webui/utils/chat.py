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
Chat utilities for SmallDoge WebUI
Handles chat completion, message formatting, and response generation
"""

import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator

from fastapi import Request
from fastapi.responses import StreamingResponse

from small_doge.webui.backend.smalldoge_webui.models.chats import (
    ChatMessage, 
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    MessageModel,
    ChatModel
)
from small_doge.webui.backend.smalldoge_webui.models.models import ModelParams
# User model removed - open access WebUI
from small_doge.webui.backend.smalldoge_webui.utils.transformers_inference import (
    create_chat_completion_response
)
from small_doge.webui.backend.smalldoge_webui.utils.models import ensure_model_loaded
from small_doge.webui.backend.smalldoge_webui.utils.models import is_model_loaded, validate_model_id
from small_doge.webui.backend.smalldoge_webui.utils.task_manager import task_manager
from small_doge.webui.backend.smalldoge_webui.constants import ERROR_MESSAGES, MESSAGE_ROLES

log = logging.getLogger(__name__)


####################
# Chat Completion Functions
####################

async def generate_chat_completion(
    request: Request,
    form_data: Dict[str, Any],
    user: Optional[Any] = None,  # User removed for open access
    bypass_filter: bool = False
) -> StreamingResponse:
    """
    Generate chat completion response with cancellation support

    Args:
        request: FastAPI request object
        form_data: Chat completion request data
        user: Optional authenticated user (None for open access)
        bypass_filter: Whether to bypass content filtering

    Returns:
        StreamingResponse: Streaming or non-streaming response
    """
    try:
        # Parse request data
        chat_request = ChatCompletionRequest(**form_data)
        
        # Validate model
        if not validate_model_id(chat_request.model):
            raise ValueError(ERROR_MESSAGES.MODEL_NOT_FOUND(chat_request.model))
        
        # Ensure model is loaded
        if not await ensure_model_loaded(chat_request.model):
            raise ValueError(ERROR_MESSAGES.MODEL_LOAD_ERROR(chat_request.model))
        
        # Create task for cancellation support
        task_id = task_manager.create_task(
            model_id=chat_request.model,
            user_id=None  # Open access - no user tracking
        )
        
        # Add task ID to request for tracking
        chat_request.task_id = task_id
        
        # Generate response
        if chat_request.stream:
            return StreamingResponse(
                stream_chat_completion_with_cancellation(chat_request, task_id),
                media_type="text/plain",
                headers={"X-Task-ID": task_id}  # Return task ID for cancellation
            )
        else:
            result = await non_stream_chat_completion_with_cancellation(chat_request, task_id)
            return result
    
    except Exception as e:
        log.error(f"Chat completion error: {e}")
        raise ValueError(ERROR_MESSAGES.MODEL_INFERENCE_ERROR(str(e)))


async def stream_chat_completion(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """
    Stream chat completion response
    
    Args:
        request: Chat completion request
    
    Yields:
        str: Server-sent events formatted response chunks
    """
    try:
        async for chunk in create_chat_completion_response(request, stream=True):
            # Format as server-sent event
            chunk_json = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {chunk_json}\n\n"
        
        # Send final event
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        log.error(f"Streaming error: {e}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "inference_error",
                "code": "model_error"
            }
        }
        error_json = json.dumps(error_chunk, ensure_ascii=False)
        yield f"data: {error_json}\n\n"


async def non_stream_chat_completion(request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Generate non-streaming chat completion response
    
    Args:
        request: Chat completion request
    
    Returns:
        Dict[str, Any]: Complete chat completion response
    """
    try:
        response = None
        async for chunk in create_chat_completion_response(request, stream=False):
            response = chunk
            break  # Non-streaming returns single response
        
        return response or {}
    
    except Exception as e:
        log.error(f"Non-streaming completion error: {e}")
        raise


####################
# Cancellable Chat Completion Functions
####################

async def stream_chat_completion_with_cancellation(
    request: ChatCompletionRequest, 
    task_id: str
) -> AsyncGenerator[str, None]:
    """
    Stream chat completion response with cancellation support
    
    Args:
        request: Chat completion request
        task_id: Task identifier for cancellation
    
    Yields:
        str: Server-sent events formatted response chunks
    """
    try:
        # Mark task as started
        task_manager.start_task(task_id)
        
        # Get cancellation event
        cancellation_event = task_manager.get_cancellation_event(task_id)
        
        # Stream with cancellation checks
        async for chunk in create_chat_completion_response_with_cancellation(
            request, task_id, stream=True
        ):
            # Check for cancellation before yielding
            if cancellation_event and cancellation_event.is_set():
                log.info(f"Task {task_id} was cancelled during streaming")
                yield "data: {\"error\": {\"message\": \"Generation cancelled by user\", \"type\": \"cancellation\", \"code\": \"user_cancelled\"}}\n\n"
                yield "data: [DONE]\n\n"
                task_manager.complete_task(task_id, error="User cancelled")
                return
            
            # Format as server-sent event
            chunk_json = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {chunk_json}\n\n"
        
        # Send final event
        yield "data: [DONE]\n\n"
        task_manager.complete_task(task_id)
    
    except Exception as e:
        log.error(f"Streaming error: {e}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "inference_error",
                "code": "model_error"
            }
        }
        error_json = json.dumps(error_chunk, ensure_ascii=False)
        yield f"data: {error_json}\n\n"
        task_manager.complete_task(task_id, error=str(e))


async def non_stream_chat_completion_with_cancellation(
    request: ChatCompletionRequest, 
    task_id: str
) -> Dict[str, Any]:
    """
    Generate non-streaming chat completion response with cancellation support
    
    Args:
        request: Chat completion request
        task_id: Task identifier for cancellation
    
    Returns:
        Dict[str, Any]: Complete chat completion response
    """
    try:
        # Mark task as started
        task_manager.start_task(task_id)
        
        response = None
        async for chunk in create_chat_completion_response_with_cancellation(
            request, task_id, stream=False
        ):
            response = chunk
            break  # Non-streaming returns single response
        
        # Check if task was cancelled
        if task_manager.is_task_cancelled(task_id):
            task_manager.complete_task(task_id, error="User cancelled")
            return {
                "error": {
                    "message": "Generation cancelled by user",
                    "type": "cancellation",
                    "code": "user_cancelled"
                }
            }
        
        task_manager.complete_task(task_id, result=response)
        return response or {}
    
    except Exception as e:
        log.error(f"Non-streaming completion error: {e}")
        task_manager.complete_task(task_id, error=str(e))
        raise


async def create_chat_completion_response_with_cancellation(
    request: ChatCompletionRequest,
    task_id: str,
    stream: bool = True
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Create chat completion response with periodic cancellation checks
    
    Args:
        request: Chat completion request
        task_id: Task identifier for cancellation
        stream: Whether to stream the response
    
    Yields:
        Dict[str, Any]: Response chunks
    """
    # Get cancellation event
    cancellation_event = task_manager.get_cancellation_event(task_id)
    
    # Use the existing inference function but with cancellation checks
    async for chunk in create_chat_completion_response(request, stream=stream):
        # Check for cancellation before yielding each chunk
        if cancellation_event and cancellation_event.is_set():
            log.info(f"Task {task_id} cancelled during generation")
            return
        
        yield chunk


####################
# Message Formatting
####################

def format_chat_messages(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """
    Format raw message data into ChatMessage objects
    
    Args:
        messages: List of raw message dictionaries
    
    Returns:
        List[ChatMessage]: Formatted chat messages
    """
    formatted_messages = []
    
    for msg in messages:
        try:
            chat_message = ChatMessage(
                role=msg.get("role", MESSAGE_ROLES.USER),
                content=msg.get("content", ""),
                name=msg.get("name"),
                function_call=msg.get("function_call"),
                tool_calls=msg.get("tool_calls")
            )
            formatted_messages.append(chat_message)
        except Exception as e:
            log.warning(f"Failed to format message: {e}")
            continue
    
    return formatted_messages


def validate_chat_messages(messages: List[Dict[str, Any]]) -> bool:
    """
    Validate chat messages format
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        bool: True if messages are valid, False otherwise
    """
    if not messages:
        return False
    
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        
        if "role" not in msg or "content" not in msg:
            return False
        
        if msg["role"] not in [MESSAGE_ROLES.SYSTEM, MESSAGE_ROLES.USER, MESSAGE_ROLES.ASSISTANT]:
            return False
        
        if not isinstance(msg["content"], str):
            return False
    
    return True


def add_system_message(messages: List[ChatMessage], system_prompt: str) -> List[ChatMessage]:
    """
    Add or update system message in chat messages
    
    Args:
        messages: List of chat messages
        system_prompt: System prompt to add
    
    Returns:
        List[ChatMessage]: Messages with system prompt
    """
    # Check if first message is already a system message
    if messages and messages[0].role == MESSAGE_ROLES.SYSTEM:
        # Update existing system message
        messages[0].content = system_prompt
        return messages
    else:
        # Add new system message at the beginning
        system_message = ChatMessage(role=MESSAGE_ROLES.SYSTEM, content=system_prompt)
        return [system_message] + messages


####################
# Response Creation
####################

def create_chat_response(
    chat_id: str,
    messages: List[MessageModel],
    model: str,
    user: Optional[Any] = None  # User removed for open access
) -> Dict[str, Any]:
    """
    Create a chat response object

    Args:
        chat_id: Chat identifier
        messages: List of messages in the chat
        model: Model used for the chat
        user: Optional user (removed for open access)

    Returns:
        Dict[str, Any]: Chat response data
    """
    return {
        "id": chat_id,
        "messages": [msg.model_dump() for msg in messages],
        "model": model,
        "created_at": int(time.time()),
        "message_count": len(messages)
    }


def create_message_response(
    message_id: str,
    chat_id: str,
    role: str,
    content: str,
    user: Optional[Any] = None,  # User removed for open access
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a message response object

    Args:
        message_id: Message identifier
        chat_id: Chat identifier
        role: Message role (user, assistant, system)
        content: Message content
        user: Optional user (removed for open access)
        model: Model used to generate the message (for assistant messages)

    Returns:
        Dict[str, Any]: Message response data
    """
    return {
        "id": message_id,
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "model": model,
        "timestamp": int(time.time()),
        "created_at": int(time.time())
    }


####################
# Chat Management
####################

def generate_chat_id() -> str:
    """Generate a unique chat ID"""
    return str(uuid.uuid4())


def generate_message_id() -> str:
    """Generate a unique message ID"""
    return str(uuid.uuid4())


def extract_chat_title(messages: List[ChatMessage], max_length: int = 50) -> str:
    """
    Extract a title for the chat based on the first user message
    
    Args:
        messages: List of chat messages
        max_length: Maximum length of the title
    
    Returns:
        str: Generated chat title
    """
    # Find the first user message
    for message in messages:
        if message.role == MESSAGE_ROLES.USER and message.content.strip():
            title = message.content.strip()
            
            # Truncate if too long
            if len(title) > max_length:
                title = title[:max_length-3] + "..."
            
            return title
    
    # Fallback title
    return "New Chat"


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count for text (rough approximation)
    
    Args:
        text: Text to count tokens for
    
    Returns:
        int: Estimated token count
    """
    # Rough estimation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def truncate_messages_by_tokens(
    messages: List[ChatMessage], 
    max_tokens: int
) -> List[ChatMessage]:
    """
    Truncate messages to fit within token limit
    
    Args:
        messages: List of chat messages
        max_tokens: Maximum token count
    
    Returns:
        List[ChatMessage]: Truncated messages
    """
    if not messages:
        return messages
    
    # Always keep system message if present
    result = []
    token_count = 0
    
    if messages[0].role == MESSAGE_ROLES.SYSTEM:
        result.append(messages[0])
        token_count += count_tokens_estimate(messages[0].content)
        messages = messages[1:]
    
    # Add messages from the end (most recent first) until token limit
    for message in reversed(messages):
        message_tokens = count_tokens_estimate(message.content)
        if token_count + message_tokens <= max_tokens:
            result.insert(-1 if result and result[0].role == MESSAGE_ROLES.SYSTEM else 0, message)
            token_count += message_tokens
        else:
            break
    
    return result
