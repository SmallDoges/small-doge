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
Chat models for SmallDoge WebUI
"""

import time
import uuid
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, String, Integer, JSON, Text, Boolean, DateTime, func
from sqlalchemy.orm import Session
from datetime import datetime

from small_doge.webui.backend.smalldoge_webui.internal.db import Base
from small_doge.webui.backend.smalldoge_webui.constants import MESSAGE_ROLES


####################
# Database Models
####################

class Chat(Base):
    __tablename__ = "chats"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)

    # Chat data (stores messages and metadata)
    chat = Column(JSON, nullable=True)

    # Chat metadata
    model = Column(String, nullable=True)
    system_prompt = Column(Text, nullable=True)

    # Chat settings
    chat_settings = Column(JSON, nullable=True)

    # Tags and categories
    tags = Column(JSON, nullable=True)
    folder_id = Column(String, nullable=True)

    # Status and sharing
    is_archived = Column(Boolean, default=False)
    is_pinned = Column(Boolean, default=False)
    is_shared = Column(Boolean, default=False)
    share_id = Column(String, nullable=True, unique=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    chat_id = Column(String, nullable=False)
    
    # Message content
    role = Column(String, nullable=False)  # system, user, assistant
    content = Column(Text, nullable=False)
    
    # Message metadata
    model = Column(String, nullable=True)
    timestamp = Column(Integer, default=lambda: int(time.time()))
    
    # Additional data
    data = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)
    
    # Message relationships
    parent_id = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()))


####################
# Pydantic Models
####################

class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatSettings(BaseModel):
    """Chat-specific settings"""
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = True
    system_prompt: Optional[str] = None


class MessageModel(BaseModel):
    """Message data model"""
    model_config = ConfigDict(from_attributes=True)

    id: str
    chat_id: str
    role: str
    content: str
    model: Optional[str] = None
    timestamp: int
    data: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    created_at: int
    updated_at: int


class ChatModel(BaseModel):
    """Chat data model"""
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    chat_settings: Optional[ChatSettings] = None
    tags: Optional[List[str]] = None
    folder_id: Optional[str] = None
    is_archived: bool = False
    is_pinned: bool = False
    is_shared: bool = False
    share_id: Optional[str] = None
    created_at: int
    updated_at: int


####################
# Response Models
####################

class MessageResponse(MessageModel):
    """Message response with additional data"""
    pass


class ChatResponse(ChatModel):
    """Chat response with messages"""
    messages: Optional[List[MessageResponse]] = None
    message_count: Optional[int] = None
    last_message_at: Optional[int] = None


class ChatListResponse(BaseModel):
    """Response for chat list endpoints"""
    chats: List[ChatResponse]
    total: int


####################
# Form Models
####################

class ChatForm(BaseModel):
    """Form for creating/updating chats"""
    title: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    chat_settings: Optional[ChatSettings] = None
    tags: Optional[List[str]] = None
    folder_id: Optional[str] = None


class ChatUpdateForm(BaseModel):
    """Form for updating chat information"""
    title: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    chat_settings: Optional[ChatSettings] = None
    tags: Optional[List[str]] = None
    folder_id: Optional[str] = None
    is_archived: Optional[bool] = None
    is_pinned: Optional[bool] = None


class MessageForm(BaseModel):
    """Form for creating messages"""
    role: str
    content: str
    model: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with cancellation support"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    task_id: Optional[str] = None  # For cancellation support


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


####################
# Utility Functions
####################

def create_chat_id() -> str:
    """Create a unique chat ID"""
    import uuid
    return str(uuid.uuid4())


def create_message_id() -> str:
    """Create a unique message ID"""
    import uuid
    return str(uuid.uuid4())


def validate_message_role(role: str) -> bool:
    """Validate message role"""
    return role in [MESSAGE_ROLES.SYSTEM, MESSAGE_ROLES.USER, MESSAGE_ROLES.ASSISTANT, MESSAGE_ROLES.FUNCTION]


def get_default_chat_settings() -> ChatSettings:
    """Get default chat settings"""
    return ChatSettings(
        model="SmallDoge/Doge-160M",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=2048,
        stream=True
    )


####################
# Database Operations Class
####################

class Chats:
    """Database operations for chats"""

    @staticmethod
    def get_chats(db: Session, limit: int = 50, skip: int = 0) -> List[Chat]:
        """Get chats with pagination"""
        return db.query(Chat).order_by(Chat.updated_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def get_chat_count(db: Session) -> int:
        """Get total chat count"""
        return db.query(Chat).count()

    @staticmethod
    def get_chat_by_id(db: Session, chat_id: str) -> Optional[Chat]:
        """Get chat by ID"""
        return db.query(Chat).filter(Chat.id == chat_id).first()

    @staticmethod
    def insert_new_chat(
        db: Session,
        chat_id: str,
        title: str,
        chat: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Optional[Chat]:
        """Insert new chat"""
        try:
            new_chat = Chat(
                id=chat_id,
                title=title,
                chat=chat,
                model=chat.get("model", "SmallDoge/Doge-160M"),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            db.add(new_chat)
            db.commit()
            db.refresh(new_chat)
            return new_chat
        except Exception as e:
            db.rollback()
            print(f"Error inserting chat: {e}")
            return None

    @staticmethod
    def update_chat_by_id(
        db: Session,
        chat_id: str,
        title: Optional[str] = None,
        chat: Optional[Dict[str, Any]] = None
    ) -> Optional[Chat]:
        """Update chat by ID"""
        try:
            chat_obj = db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat_obj:
                return None

            if title is not None:
                chat_obj.title = title
            if chat is not None:
                chat_obj.chat = chat
                chat_obj.model = chat.get("model", chat_obj.model)

            chat_obj.updated_at = datetime.now()
            db.commit()
            db.refresh(chat_obj)
            return chat_obj
        except Exception as e:
            db.rollback()
            print(f"Error updating chat: {e}")
            return None

    @staticmethod
    def delete_chat_by_id(db: Session, chat_id: str) -> bool:
        """Delete chat by ID"""
        try:
            chat_obj = db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat_obj:
                return False

            db.delete(chat_obj)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Error deleting chat: {e}")
            return False

    @staticmethod
    def get_chats_by_user_id(db: Session, user_id: str, limit: int = 50, skip: int = 0) -> List[Chat]:
        """Get chats by user ID (for future use if authentication is added)"""
        # For now, return all chats since we don't have user authentication
        return Chats.get_chats(db, limit, skip)

    @staticmethod
    def search_chats(db: Session, query: str, limit: int = 50, skip: int = 0) -> List[Chat]:
        """Search chats by title or content"""
        return db.query(Chat).filter(
            Chat.title.contains(query)
        ).order_by(Chat.updated_at.desc()).offset(skip).limit(limit).all()
