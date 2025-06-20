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
Model management router for SmallDoge WebUI
"""

import logging
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
import psutil
import torch
import time

from small_doge.webui.backend.smalldoge_webui.models.models import ModelResponse, ModelListResponse
# Authentication removed for open source sharing
from small_doge.webui.backend.smalldoge_webui.utils.models import (
    get_available_models,
    get_loaded_models,
    get_model_status,
    load_model,
    unload_model,
    health_check_model,
    get_model_capabilities,
    get_model_context_length,
    get_model_info
)
from small_doge.webui.backend.smalldoge_webui.constants import ERROR_MESSAGES

log = logging.getLogger(__name__)
router = APIRouter()


####################
# Model Information
####################

@router.get("/", response_model=List[str])
async def get_models():
    """
    Get available models

    Returns:
        List[str]: List of available model IDs
    """
    try:
        return get_available_models()
    except Exception as e:
        log.error(f"Error getting models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/loaded")
async def get_loaded_models_endpoint():
    """
    Get currently loaded models

    Returns:
        List[str]: List of loaded model IDs
    """
    try:
        return {"loaded_models": get_loaded_models()}
    except Exception as e:
        log.error(f"Error getting loaded models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/{model_id:path}/status")
async def get_model_status_endpoint(model_id: str):
    """
    Get model status

    Args:
        model_id: Model identifier

    Returns:
        dict: Model status information
    """
    try:
        return get_model_status(model_id)
    except Exception as e:
        log.error(f"Error getting model status for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Model Management
####################

@router.post("/{model_id:path}/load")
async def load_model_endpoint(model_id: str):
    """
    Load a model

    Args:
        model_id: Model identifier

    Returns:
        dict: Success message
    """
    try:
        available_models = get_available_models()
        if model_id not in available_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(model_id)
            )
        
        success = await load_model(model_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERROR_MESSAGES.MODEL_LOAD_ERROR(model_id)
            )
        
        log.info(f"Model {model_id} loaded successfully")
        return {"message": f"Model {model_id} loaded successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error loading model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.MODEL_LOAD_ERROR(model_id, str(e))
        )


@router.post("/{model_id:path}/unload")
async def unload_model_endpoint(model_id: str):
    """
    Unload a model

    Args:
        model_id: Model identifier

    Returns:
        dict: Success message
    """
    try:
        success = await unload_model(model_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to unload model {model_id}"
            )
        
        log.info(f"Model {model_id} unloaded successfully")
        return {"message": f"Model {model_id} unloaded successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error unloading model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.delete("/{model_id:path}/remove")
async def remove_model_endpoint(model_id: str):
    """
    Remove a model from available models list and unload it from memory

    Args:
        model_id: Model identifier

    Returns:
        dict: Operation results
    """
    try:
        from small_doge.webui.backend.smalldoge_webui.utils.models import unload_and_remove_model
        
        result = await unload_and_remove_model(model_id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", f"Failed to remove model {model_id}")
            )
        
        log.info(f"Model {model_id} removed successfully")
        return {
            "message": f"Model {model_id} removed successfully",
            "details": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error removing model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Model Health
####################

@router.get("/{model_id:path}/health")
async def health_check_model_endpoint(model_id: str):
    """
    Perform health check on a model

    Args:
        model_id: Model identifier

    Returns:
        dict: Health check results
    """
    try:
        return await health_check_model(model_id)
    except Exception as e:
        log.error(f"Error performing health check for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Enhanced Model Information (Open-WebUI Compatible)
####################

@router.get("/{model_id:path}/info")
async def get_model_info_endpoint(model_id: str):
    """
    Get detailed model information

    Args:
        model_id: Model identifier

    Returns:
        dict: Detailed model information
    """
    try:
        available_models = get_available_models()
        if model_id not in available_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(model_id)
            )

        model_info = get_model_info(model_id)
        capabilities = get_model_capabilities(model_id)
        context_length = get_model_context_length(model_id)
        status_info = get_model_status(model_id)

        return {
            "id": model_id,
            "name": model_id,
            "info": model_info,
            "capabilities": capabilities,
            "context_length": context_length,
            "status": status_info,
            "supports_streaming": True,
            "supports_functions": False,
            "supports_vision": False,
            "supports_tools": False,
            "owned_by": "smalldoge-webui",
            "created": status_info.get("loaded_at", 0)
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting model info for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/{model_id:path}/capabilities")
async def get_model_capabilities_endpoint(model_id: str):
    """
    Get model capabilities (Open-WebUI compatible)

    Args:
        model_id: Model identifier

    Returns:
        dict: Model capabilities
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
            "max_tokens": context_length if isinstance(context_length, int) else 2048,
            "supports_streaming": True,
            "supports_functions": False,
            "supports_vision": False,
            "supports_tools": False,
            "supports_system_messages": True,
            "supports_user_messages": True,
            "supports_assistant_messages": True
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
# Model Statistics and Metrics
####################

@router.get("/stats")
async def get_models_stats():
    """
    Get overall model statistics

    Returns:
        dict: Model statistics
    """
    try:
        available_models = get_available_models()
        loaded_models = get_loaded_models()

        stats = {
            "total_available": len(available_models),
            "total_loaded": len(loaded_models),
            "available_models": available_models,
            "loaded_models": loaded_models,
            "memory_usage": {},
            "inference_stats": {}
        }

        # Get individual model stats
        for model_id in loaded_models:
            try:
                model_status = get_model_status(model_id)
                stats["memory_usage"][model_id] = model_status.get("memory_usage", "unknown")
                stats["inference_stats"][model_id] = model_status.get("inference_count", 0)
            except Exception as e:
                log.warning(f"Could not get stats for model {model_id}: {e}")

        return stats

    except Exception as e:
        log.error(f"Error getting model stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.post("/reload-all")
async def reload_all_models():
    """
    Reload all currently loaded models

    Returns:
        dict: Reload results
    """
    try:
        loaded_models = get_loaded_models()
        results = {}

        for model_id in loaded_models:
            try:
                # Unload first
                await unload_model(model_id)
                # Then reload
                success = await load_model(model_id)
                results[model_id] = "success" if success else "failed"
            except Exception as e:
                results[model_id] = f"error: {str(e)}"

        return {
            "message": "Model reload completed",
            "results": results,
            "total_processed": len(loaded_models)
        }

    except Exception as e:
        log.error(f"Error reloading models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/performance")
async def get_performance_stats() -> Dict[str, Any]:
    """获取系统性能统计信息"""
    try:
        # 获取CPU使用率（使用interval=0.1来获取更准确的瞬时值）
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 获取GPU信息（如果可用）
        gpu_info = {
            "available": False,
            "name": "CPU",
            "usage": cpu_percent,
            "memory_used": 0,
            "memory_total": 0
        }
        
        if torch.cuda.is_available():
            try:
                # 获取GPU设备名称
                gpu_info["available"] = True
                gpu_info["name"] = torch.cuda.get_device_name(0)
                
                # 尝试获取GPU使用率和内存信息
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # 获取GPU使用率
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info["usage"] = utilization.gpu
                    
                    # 获取显存使用情况
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info["memory_used"] = memory.used // 1024 // 1024  # Convert to MB
                    gpu_info["memory_total"] = memory.total // 1024 // 1024  # Convert to MB
                    
                    # 获取功耗信息（如果支持）
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
                        gpu_info["power_usage"] = f"{power:.1f}W"
                    except:
                        gpu_info["power_usage"] = "N/A"
                        
                except Exception as e:
                    print(f"Error getting NVIDIA GPU stats: {e}")
                    # 如果无法获取详细GPU信息，使用PyTorch的基本信息
                    gpu_info["usage"] = 0
                    gpu_info["memory_used"] = torch.cuda.memory_allocated(0) // 1024 // 1024
                    gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
            except Exception as e:
                print(f"Error accessing GPU: {e}")
                gpu_info["available"] = False
        
        # 获取系统内存使用情况
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total // 1024 // 1024,  # MB
            "used": memory.used // 1024 // 1024,  # MB
            "percent": memory.percent
        }
        
        # 获取进程信息
        process = psutil.Process()
        process_info = {
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_percent": process.memory_percent(),
            "threads": process.num_threads()
        }
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "gpu": gpu_info,
            "memory": memory_info,
            "process": process_info
        }
    except Exception as e:
        print(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
