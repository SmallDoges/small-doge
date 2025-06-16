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
Configuration management for SmallDoge WebUI
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Generic, Optional, TypeVar, Dict, Any

from pydantic import BaseModel
from sqlalchemy import JSON, Column, DateTime, Integer, func

from smalldoge_webui.env import (
    DATA_DIR,
    DATABASE_URL,
    ENV,
    WEBUI_AUTH,
    WEBUI_NAME,
    WEBUI_FAVICON_URL,
    WEBUI_SECRET_KEY,
    DEFAULT_MODELS,
    log,
)
from smalldoge_webui.internal.db import Base, get_db_context


####################
# Database Model
####################

class Config(Base):
    __tablename__ = "config"

    id = Column(Integer, primary_key=True)
    data = Column(JSON, nullable=False)
    version = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())


####################
# Configuration Management
####################

def load_json_config() -> Dict[str, Any]:
    """Load configuration from JSON file"""
    config_file = DATA_DIR / "config.json"
    if config_file.exists():
        with open(config_file, "r") as file:
            return json.load(file)
    return {}


def save_to_db(data: Dict[str, Any]) -> bool:
    """Save configuration to database"""
    try:
        with get_db_context() as db:
            existing_config = db.query(Config).first()
            if not existing_config:
                new_config = Config(data=data, version=0)
                db.add(new_config)
            else:
                existing_config.data = data
                existing_config.updated_at = datetime.now()
                db.add(existing_config)
        return True
    except Exception as e:
        log.error(f"Failed to save config to database: {e}")
        return False


def reset_config() -> bool:
    """Reset configuration to defaults"""
    try:
        with get_db_context() as db:
            db.query(Config).delete()
        return True
    except Exception as e:
        log.error(f"Failed to reset config: {e}")
        return False


# Default configuration
DEFAULT_CONFIG = {
    "version": 0,
    "ui": {
        "name": WEBUI_NAME,
        "favicon_url": WEBUI_FAVICON_URL,
        "theme": "light",
        "language": "en",
        "show_username": True,
        "show_timestamp": True,
    },
    "auth": {
        "enabled": False,  # Disabled for open source sharing
        "signup_enabled": False,
        "default_role": "user",
        "jwt_expires_in": "7d",
    },
    "models": {
        "default": DEFAULT_MODELS,
        "available": [DEFAULT_MODELS],
        "auto_load": True,
        "cache_enabled": True,
    },
    "chat": {
        "max_history": 100,
        "default_system_prompt": "You are a helpful AI assistant.",
        "enable_streaming": True,
        "enable_web_search": False,
    },
    "inference": {
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "device": "auto",
        "torch_dtype": "auto",
    },
    "security": {
        "content_filter_enabled": False,
        "rate_limit_enabled": True,
        "max_requests_per_hour": 100,
    },
    "features": {
        "file_upload_enabled": True,
        "image_generation_enabled": False,
        "voice_enabled": False,
        "plugins_enabled": False,
    }
}


def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    try:
        with get_db_context() as db:
            config_entry = db.query(Config).order_by(Config.id.desc()).first()
            if config_entry:
                return config_entry.data
    except Exception as e:
        log.error(f"Failed to get config from database: {e}")
    
    return DEFAULT_CONFIG.copy()


# Load initial configuration
CONFIG_DATA = get_config()

# Migrate JSON config to database if exists
config_file = DATA_DIR / "config.json"
if config_file.exists():
    try:
        json_config = load_json_config()
        if json_config:
            save_to_db(json_config)
            # Backup and remove JSON file
            backup_file = DATA_DIR / "old_config.json"
            config_file.rename(backup_file)
            log.info("Migrated JSON config to database")
    except Exception as e:
        log.error(f"Failed to migrate JSON config: {e}")


def get_config_value(config_path: str) -> Any:
    """Get configuration value by path (e.g., 'ui.theme')"""
    path_parts = config_path.split(".")
    current_config = CONFIG_DATA
    
    for key in path_parts:
        if isinstance(current_config, dict) and key in current_config:
            current_config = current_config[key]
        else:
            return None
    
    return current_config


def set_config_value(config_path: str, value: Any) -> bool:
    """Set configuration value by path"""
    global CONFIG_DATA
    
    path_parts = config_path.split(".")
    current_config = CONFIG_DATA
    
    # Navigate to the parent of the target key
    for key in path_parts[:-1]:
        if key not in current_config:
            current_config[key] = {}
        current_config = current_config[key]
    
    # Set the value
    current_config[path_parts[-1]] = value
    
    # Save to database
    return save_config(CONFIG_DATA)


def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration"""
    global CONFIG_DATA
    
    try:
        if save_to_db(config):
            CONFIG_DATA = config
            log.info("Configuration saved successfully")
            return True
    except Exception as e:
        log.error(f"Failed to save configuration: {e}")
    
    return False


####################
# Persistent Configuration
####################

T = TypeVar("T")

ENABLE_PERSISTENT_CONFIG = (
    os.environ.get("ENABLE_PERSISTENT_CONFIG", "True").lower() == "true"
)

PERSISTENT_CONFIG_REGISTRY = []


class PersistentConfig(Generic[T]):
    """Persistent configuration value that can be updated from database"""
    
    def __init__(self, env_name: str, config_path: str, env_value: T):
        self.env_name = env_name
        self.config_path = config_path
        self.env_value = env_value
        self.config_value = get_config_value(config_path)
        
        if self.config_value is not None and ENABLE_PERSISTENT_CONFIG:
            log.info(f"'{env_name}' loaded from database configuration")
            self.value = self.config_value
        else:
            self.value = env_value
        
        PERSISTENT_CONFIG_REGISTRY.append(self)
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return f"PersistentConfig({self.env_name}={self.value})"
    
    def update(self):
        """Update value from database"""
        new_value = get_config_value(self.config_path)
        if new_value is not None:
            self.value = new_value
            log.info(f"Updated {self.env_name} to new value: {self.value}")
    
    def save(self):
        """Save current value to database"""
        log.info(f"Saving '{self.env_name}' to database")
        if set_config_value(self.config_path, self.value):
            self.config_value = self.value
            return True
        return False


####################
# Application Configuration
####################

class AppConfig:
    """Application configuration manager"""
    
    def __init__(self):
        self._config = CONFIG_DATA.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return get_config_value(key) or default
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        return set_config_value(key, value)
    
    def update(self, config: Dict[str, Any]) -> bool:
        """Update multiple configuration values"""
        try:
            # Merge with existing config
            updated_config = CONFIG_DATA.copy()
            self._deep_update(updated_config, config)
            return save_config(updated_config)
        except Exception as e:
            log.error(f"Failed to update configuration: {e}")
            return False
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def reset(self) -> bool:
        """Reset configuration to defaults"""
        return save_config(DEFAULT_CONFIG.copy())
    
    def export(self) -> Dict[str, Any]:
        """Export current configuration"""
        return CONFIG_DATA.copy()
    
    def import_config(self, config: Dict[str, Any]) -> bool:
        """Import configuration"""
        return save_config(config)


# Global configuration instance
app_config = AppConfig()
