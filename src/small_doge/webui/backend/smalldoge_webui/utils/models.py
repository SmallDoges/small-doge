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
Model management utilities for SmallDoge WebUI
Enhanced with HuggingFace Hub integration
"""

import logging
from typing import List, Dict, Any, Optional, Union

from smalldoge_webui.utils.transformers_inference import (
    model_manager,
    load_model as _load_model,
    unload_model as _unload_model,
    is_model_loaded as _is_model_loaded,
    get_loaded_models as _get_loaded_models,
    get_available_models as _get_available_models,
    get_model_info as _get_model_info,
)
from smalldoge_webui.utils.huggingface_integration import (
    search_huggingface_models,
    get_task_categories,
    check_model_compatibility_public,
    get_model_families_public
)

log = logging.getLogger(__name__)


####################
# Model Management Functions
####################

async def load_model(model_id: str, **kwargs) -> bool:
    """
    Load a model for inference
    
    Args:
        model_id: Model identifier (e.g., "SmallDoge/Doge-160M")
        **kwargs: Additional loading parameters
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    try:
        log.info(f"Loading model: {model_id}")
        result = await _load_model(model_id, **kwargs)
        if result:
            log.info(f"Model {model_id} loaded successfully")
        else:
            log.error(f"Failed to load model {model_id}")
        return result
    except Exception as e:
        log.error(f"Error loading model {model_id}: {e}")
        return False


async def unload_model(model_id: str) -> bool:
    """
    Unload a model from memory
    
    Args:
        model_id: Model identifier
    
    Returns:
        bool: True if model unloaded successfully, False otherwise
    """
    try:
        log.info(f"Unloading model: {model_id}")
        result = await _unload_model(model_id)
        if result:
            log.info(f"Model {model_id} unloaded successfully")
        else:
            log.error(f"Failed to unload model {model_id}")
        return result
    except Exception as e:
        log.error(f"Error unloading model {model_id}: {e}")
        return False


def is_model_loaded(model_id: str) -> bool:
    """
    Check if a model is currently loaded
    
    Args:
        model_id: Model identifier
    
    Returns:
        bool: True if model is loaded, False otherwise
    """
    return _is_model_loaded(model_id)


def get_loaded_models() -> List[str]:
    """
    Get list of currently loaded models
    
    Returns:
        List[str]: List of loaded model identifiers
    """
    return _get_loaded_models()


def get_available_models() -> List[str]:
    """
    Get list of available models
    
    Returns:
        List[str]: List of available model identifiers
    """
    # Start with base SmallDoge models
    base_models = _get_available_models()
    
    # Add all currently loaded models (including dynamically loaded HuggingFace models)
    loaded_models = get_loaded_models()
    
    # Combine and deduplicate
    all_models = list(set(base_models + loaded_models))
    
    return all_models


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        Optional[Dict[str, Any]]: Model information or None if not found
    """
    return _get_model_info(model_id)


def get_model_status(model_id: str) -> Dict[str, Any]:
    """
    Get detailed status of a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        Dict[str, Any]: Model status information
    """
    import time
    
    status = {
        "id": model_id,
        "status": "unknown",
        "is_loaded": is_model_loaded(model_id),
        "is_available": model_id in get_available_models(),
        "timestamp": int(time.time())
    }
    
    if status["is_loaded"]:
        info = get_model_info(model_id)
        if info:
            status.update({
                "status": "loaded",
                "load_time": info.get("load_time"),
                "device": info.get("device"),
                "torch_dtype": info.get("torch_dtype"),
                "memory_usage": "unknown",  # TODO: Implement actual memory tracking
                "inference_count": 0  # TODO: Implement actual inference counting
            })
        else:
            status["status"] = "error"
    elif status["is_available"]:
        status["status"] = "available"
    else:
        status["status"] = "unavailable"
    
    return status


async def load_default_model() -> bool:
    """
    Load the default model
    
    Returns:
        bool: True if default model loaded successfully, False otherwise
    """
    from smalldoge_webui.env import DEFAULT_MODELS
    
    default_model = DEFAULT_MODELS.split(",")[0].strip()
    log.info(f"Loading default model: {default_model}")
    
    return await load_model(default_model)


async def ensure_model_loaded(model_id: str) -> bool:
    """
    Ensure a model is loaded, loading it if necessary

    Args:
        model_id: Model identifier

    Returns:
        bool: True if model is loaded, False if loading failed
    """
    if is_model_loaded(model_id):
        return True

    return await load_model(model_id)


def is_huggingface_model_id(model_id: str) -> bool:
    """
    Check if a model ID is a HuggingFace model ID (org/model format)
    
    Args:
        model_id: Model identifier
    
    Returns:
        bool: True if it's a HuggingFace model ID, False if it's a local path
    """
    # Count slashes
    slash_count = model_id.count('/')
    
    # HuggingFace model IDs have exactly one slash (org/model)
    if slash_count == 1:
        # Additional validation: should not start with / or contain backslashes
        if not model_id.startswith('/') and '\\' not in model_id:
            parts = model_id.split('/')
            # Both parts should be non-empty and not contain special path characters
            if len(parts) == 2 and all(part.strip() for part in parts):
                return True
    
    return False


def is_local_model_path(model_id: str) -> bool:
    """
    Check if a model ID is a local file system path
    
    Args:
        model_id: Model identifier
    
    Returns:
        bool: True if it's a local path, False otherwise
    """
    # Multiple slashes or backslashes indicate local path
    if model_id.count('/') > 1 or '\\' in model_id:
        return True
    
    # Starts with / (Unix) or C:\ (Windows) patterns
    if model_id.startswith('/') or (len(model_id) > 2 and model_id[1:3] == ':\\'):
        return True
    
    # Relative paths starting with ./ or ../
    if model_id.startswith('./') or model_id.startswith('../'):
        return True
    
    return False


def validate_model_id(model_id: str) -> bool:
    """
    Validate if a model ID is valid and available
    
    Args:
        model_id: Model identifier to validate
    
    Returns:
        bool: True if model ID is valid, False otherwise
    """
    # Check if model is in available models list
    available_models = get_available_models()
    if model_id in available_models:
        return True
    
    # Check if model is already loaded (for dynamically loaded models)
    if is_model_loaded(model_id):
        return True
    
    # Validate HuggingFace model ID format
    if is_huggingface_model_id(model_id):
        return True
    
    # Validate local model path (basic check - could be enhanced)
    if is_local_model_path(model_id):
        # For local paths, we could add additional validation like checking if path exists
        # For now, we'll allow it and let the loading process handle validation
        return True
    
    return False


def get_model_capabilities(model_id: str) -> List[str]:
    """
    Get capabilities of a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        List[str]: List of model capabilities
    """
    # Default capabilities for all models
    capabilities = ["text-generation", "chat-completion"]
    
    # Add model-specific capabilities based on model ID
    if "doge" in model_id.lower():
        capabilities.extend(["conversation", "instruction-following"])
    
    return capabilities


def get_model_context_length(model_id: str) -> Union[int, str]:
    """
    Get context length for a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        Union[int, str]: Context length in tokens, or "unknown" if not available
    """
    # First try to get from loaded model configuration
    model_info = get_model_info(model_id)
    if model_info and "config" in model_info:
        config = model_info["config"]
        if "context_length" in config and config["context_length"]:
            return config["context_length"]
        if "max_position_embeddings" in config and config["max_position_embeddings"]:
            return config["max_position_embeddings"]
    
    # If no configuration available, return "unknown"
    return "unknown"


####################
# Model Health Checks
####################

async def health_check_model(model_id: str) -> Dict[str, Any]:
    """
    Perform health check on a model
    
    Args:
        model_id: Model identifier
    
    Returns:
        Dict[str, Any]: Health check results
    """
    import time
    
    health_status = {
        "model_id": model_id,
        "status": "unknown",
        "message": "",
        "timestamp": int(time.time()),
        "response_time": None,
    }
    
    try:
        start_time = time.time()
        
        # Check if model is loaded
        if not is_model_loaded(model_id):
            health_status.update({
                "status": "not_loaded",
                "message": "Model is not loaded"
            })
            return health_status
        
        # Try a simple inference
        from smalldoge_webui.models.chats import ChatMessage
        from smalldoge_webui.utils.transformers_inference import generate_completion
        
        test_messages = [
            ChatMessage(role="user", content="Hello")
        ]
        
        # Generate a short response to test the model
        response_generated = False
        async for token in generate_completion(model_id, test_messages, stream=True):
            if token.strip():
                response_generated = True
                break
        
        response_time = time.time() - start_time
        
        if response_generated:
            health_status.update({
                "status": "healthy",
                "message": "Model is responding normally",
                "response_time": response_time
            })
        else:
            health_status.update({
                "status": "unhealthy",
                "message": "Model did not generate response",
                "response_time": response_time
            })
    
    except Exception as e:
        health_status.update({
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        })
    
    return health_status


async def health_check_all_models() -> Dict[str, Dict[str, Any]]:
    """
    Perform health check on all loaded models
    
    Returns:
        Dict[str, Dict[str, Any]]: Health check results for all models
    """
    results = {}
    loaded_models = get_loaded_models()
    
    for model_id in loaded_models:
        results[model_id] = await health_check_model(model_id)
    
    return results


####################
# HuggingFace Integration Functions
####################

def search_models_by_task(task: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search HuggingFace models by task
    
    Args:
        task: Task category (e.g., "text-generation", "conversational")
        limit: Maximum number of results
    
    Returns:
        List[Dict[str, Any]]: List of model information
    """
    try:
        return search_huggingface_models(task=task, limit=limit)
    except Exception as e:
        log.error(f"Error searching models by task {task}: {e}")
        return []





def get_available_task_categories() -> List[Dict[str, str]]:
    """
    Get available task categories for model search
    
    Returns:
        List[Dict[str, str]]: List of task categories with value and label
    """
    try:
        return get_task_categories()
    except Exception as e:
        log.error(f"Error getting task categories: {e}")
        return []


def check_huggingface_model_compatibility(model_id: str) -> Dict[str, Any]:
    """
    Check if a HuggingFace model is compatible with our system
    
    Args:
        model_id: HuggingFace model identifier
    
    Returns:
        Dict[str, Any]: Compatibility information
    """
    try:
        return check_model_compatibility_public(model_id)
    except Exception as e:
        log.error(f"Error checking compatibility for {model_id}: {e}")
        return {
            "model_id": model_id,
            "compatible": False,
            "error": str(e)
        }


def get_model_families() -> Dict[str, List[str]]:
    """
    Get organized model families for easy browsing
    
    Returns:
        Dict[str, List[str]]: Model families organized by category
    """
    try:
        return get_model_families_public()
    except Exception as e:
        log.error(f"Error getting model families: {e}")
        return {"SmallDoge": ["SmallDoge/Doge-160M"]}


def get_popular_models_by_category(category: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get popular models by category
    
    Args:
        category: Model category
        limit: Maximum number of results
    
    Returns:
        List[Dict[str, Any]]: List of popular models
    """
    try:
        # Map categories to search parameters
        category_mapping = {
            "chat": {"task": "conversational"},
            "code": {"tags": ["code", "programming"]},
            "instruct": {"tags": ["instruct", "instruction"]},
            "small": {"query": "small", "task": "text-generation"},
            "multilingual": {"tags": ["multilingual"]},
            "text-generation": {"task": "text-generation"}
        }
        
        search_params = category_mapping.get(category, {"task": "text-generation"})
        return search_huggingface_models(limit=limit, **search_params)
        
    except Exception as e:
        log.error(f"Error getting popular models for category {category}: {e}")
        return []


async def validate_and_load_huggingface_model(model_id: str) -> Dict[str, Any]:
    """
    Validate a HuggingFace model and attempt to load it
    
    Args:
        model_id: HuggingFace model identifier
    
    Returns:
        Dict[str, Any]: Validation and loading results
    """
    try:
        # First check compatibility
        compatibility = check_huggingface_model_compatibility(model_id)
        
        result = {
            "model_id": model_id,
            "compatibility": compatibility,
            "loaded": False,
            "success": False,
            "error": None
        }
        
        if not compatibility.get("compatible", False):
            issues = compatibility.get("issues", [])
            warnings = compatibility.get("warnings", [])
            error_details = []
            if issues:
                error_details.extend(issues)
            if warnings:
                error_details.extend(warnings)
            result["error"] = f"Model compatibility issues: {'; '.join(error_details)}"
            return result
        
        # Try to load the model
        log.info(f"Attempting to load HuggingFace model: {model_id}")
        success = await load_model(model_id)
        
        if success:
            result["loaded"] = True
            result["success"] = True
            log.info(f"Successfully loaded HuggingFace model: {model_id}")
            
            # Add to available models list if not already present
            available_models = get_available_models()
            if model_id not in available_models:
                # This is a bit of a hack - we'll modify the global list
                from smalldoge_webui.constants import MODEL_CONFIG
                if model_id not in MODEL_CONFIG.SMALLDOGE_MODELS:
                    MODEL_CONFIG.SMALLDOGE_MODELS.append(model_id)
                    log.info(f"Added {model_id} to available models list")
        else:
            result["error"] = "Failed to load model - check logs for details"
            log.error(f"Failed to load HuggingFace model: {model_id}")
        
        return result
        
    except Exception as e:
        log.error(f"Error validating and loading model {model_id}: {e}")
        return {
            "model_id": model_id,
            "loaded": False,
            "success": False,
            "error": str(e)
        }


def remove_model_from_available(model_id: str) -> bool:
    """
    Remove a model from the available models list
    
    Args:
        model_id: Model identifier to remove
    
    Returns:
        bool: True if model was removed, False if not found
    """
    try:
        from smalldoge_webui.constants import MODEL_CONFIG
        
        if model_id in MODEL_CONFIG.SMALLDOGE_MODELS:
            MODEL_CONFIG.SMALLDOGE_MODELS.remove(model_id)
            log.info(f"Removed {model_id} from available models list")
            return True
        else:
            log.warning(f"Model {model_id} not found in available models list")
            return False
            
    except Exception as e:
        log.error(f"Error removing model {model_id} from available list: {e}")
        return False


async def unload_and_remove_model(model_id: str) -> Dict[str, Any]:
    """
    Unload a model from memory and remove it from available models list
    
    Args:
        model_id: Model identifier
    
    Returns:
        Dict[str, Any]: Operation results
    """
    try:
        result = {
            "model_id": model_id,
            "unloaded": False,
            "removed_from_list": False,
            "success": False,
            "error": None
        }
        
        # First unload the model if it's loaded
        if is_model_loaded(model_id):
            unload_success = await unload_model(model_id)
            result["unloaded"] = unload_success
            if not unload_success:
                result["error"] = "Failed to unload model from memory"
                return result
        else:
            result["unloaded"] = True  # Not loaded, so "unloading" is successful
        
        # Remove from available models list
        remove_success = remove_model_from_available(model_id)
        result["removed_from_list"] = remove_success
        
        if remove_success:
            result["success"] = True
            log.info(f"Successfully unloaded and removed model: {model_id}")
        else:
            result["error"] = "Model was unloaded but could not be removed from available list"
        
        return result
        
    except Exception as e:
        log.error(f"Error unloading and removing model {model_id}: {e}")
        return {
            "model_id": model_id,
            "unloaded": False,
            "removed_from_list": False,
            "success": False,
            "error": str(e)
        }
