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
Chat management router for SmallDoge WebUI
Open source feature sharing - no authentication required
Enhanced with proper chat session management
"""

import logging
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel

from small_doge.webui.backend.smalldoge_webui.constants import ERROR_MESSAGES
from small_doge.webui.backend.smalldoge_webui.internal.db import get_db
from small_doge.webui.backend.smalldoge_webui.models.chats import Chats

log = logging.getLogger(__name__)
router = APIRouter()


####################
# Pydantic Models
####################

class ChatCreateRequest(BaseModel):
    title: Optional[str] = None
    model: str = "SmallDoge/Doge-160M"
    messages: List[Dict[str, Any]] = []

class ChatUpdateRequest(BaseModel):
    title: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    id: str
    title: str
    model: str
    messages: List[Dict[str, Any]]
    created_at: int
    updated_at: int
    message_count: int


####################
# Chat Management Endpoints
####################

@router.get("/")
async def get_chats(limit: int = 50, offset: int = 0):
    """
    Get chats (open access)

    Args:
        limit: Maximum number of chats to return
        offset: Number of chats to skip

    Returns:
        Dict: List of chats with pagination info
    """
    try:
        db = next(get_db())
        chats = Chats.get_chats(db, limit=limit, skip=offset)
        total = Chats.get_chat_count(db)

        chat_list = []
        for chat in chats:
            chat_data = {
                "id": chat.id,
                "title": chat.title,
                "model": chat.model,
                "created_at": int(chat.created_at.timestamp()) if chat.created_at else 0,
                "updated_at": int(chat.updated_at.timestamp()) if chat.updated_at else 0,
                "message_count": len(chat.chat.get("messages", [])) if chat.chat else 0
            }
            chat_list.append(chat_data)

        return {
            "chats": chat_list,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(chat_list) < total
        }

    except Exception as e:
        log.error(f"Error getting chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.post("/", response_model=ChatResponse)
async def create_chat(chat_request: ChatCreateRequest):
    """
    Create a new chat (open access)

    Args:
        chat_request: Chat creation request

    Returns:
        ChatResponse: Created chat information
    """
    try:
        db = next(get_db())

        # Generate chat ID
        chat_id = str(uuid.uuid4())

        # Create chat title if not provided
        title = chat_request.title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # Prepare chat data
        chat_data = {
            "messages": chat_request.messages,
            "model": chat_request.model,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # Create chat in database
        chat = Chats.insert_new_chat(
            db,
            chat_id=chat_id,
            title=title,
            chat=chat_data,
            user_id=None  # No user authentication
        )

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create chat"
            )

        return ChatResponse(
            id=chat.id,
            title=chat.title,
            model=chat_request.model,
            messages=chat_request.messages,
            created_at=int(chat.created_at.timestamp()) if chat.created_at else int(time.time()),
            updated_at=int(chat.updated_at.timestamp()) if chat.updated_at else int(time.time()),
            message_count=len(chat_request.messages)
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error creating chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(chat_id: str):
    """
    Get specific chat (open access)

    Args:
        chat_id: Chat identifier

    Returns:
        ChatResponse: Chat information
    """
    try:
        db = next(get_db())
        chat = Chats.get_chat_by_id(db, chat_id)

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat {chat_id} not found"
            )

        chat_data = chat.chat or {}
        messages = chat_data.get("messages", [])
        model = chat_data.get("model", "SmallDoge/Doge-160M")

        return ChatResponse(
            id=chat.id,
            title=chat.title,
            model=model,
            messages=messages,
            created_at=int(chat.created_at.timestamp()) if chat.created_at else 0,
            updated_at=int(chat.updated_at.timestamp()) if chat.updated_at else 0,
            message_count=len(messages)
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.put("/{chat_id}", response_model=ChatResponse)
async def update_chat(chat_id: str, chat_update: ChatUpdateRequest):
    """
    Update chat (open access)

    Args:
        chat_id: Chat identifier
        chat_update: Chat update request

    Returns:
        ChatResponse: Updated chat information
    """
    try:
        db = next(get_db())
        chat = Chats.get_chat_by_id(db, chat_id)

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat {chat_id} not found"
            )

        # Update chat data
        chat_data = chat.chat or {}

        if chat_update.messages is not None:
            chat_data["messages"] = chat_update.messages

        chat_data["updated_at"] = datetime.now().isoformat()

        # Update title if provided
        title = chat_update.title if chat_update.title is not None else chat.title

        # Update in database
        updated_chat = Chats.update_chat_by_id(
            db,
            chat_id=chat_id,
            title=title,
            chat=chat_data
        )

        if not updated_chat:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update chat"
            )

        messages = chat_data.get("messages", [])
        model = chat_data.get("model", "SmallDoge/Doge-160M")

        return ChatResponse(
            id=updated_chat.id,
            title=updated_chat.title,
            model=model,
            messages=messages,
            created_at=int(updated_chat.created_at.timestamp()) if updated_chat.created_at else 0,
            updated_at=int(updated_chat.updated_at.timestamp()) if updated_chat.updated_at else 0,
            message_count=len(messages)
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error updating chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str):
    """
    Delete chat (open access)

    Args:
        chat_id: Chat identifier

    Returns:
        Dict: Deletion confirmation
    """
    try:
        db = next(get_db())
        chat = Chats.get_chat_by_id(db, chat_id)

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat {chat_id} not found"
            )

        # Delete chat
        success = Chats.delete_chat_by_id(db, chat_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete chat"
            )

        return {
            "message": f"Chat {chat_id} deleted successfully",
            "chat_id": chat_id,
            "deleted_at": int(time.time())
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error deleting chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )
