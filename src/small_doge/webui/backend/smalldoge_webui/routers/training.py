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
Training management router for SmallDoge WebUI
Provides API endpoints for model training management
"""

import logging
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel

from smalldoge_webui.utils.training_utils import (
    get_available_training_types,
    get_available_models,
    get_training_configs,
    start_training,
    get_training_status,
    list_training_jobs,
    get_training_logs,
    stop_training
)
from smalldoge_webui.constants import ERROR_MESSAGES

logger = logging.getLogger(__name__)

router = APIRouter()


####################
# Request Models
####################

class TrainingStartRequest(BaseModel):
    training_type: str  # pretrain, sft, dpo, grpo
    model_type: str     # doge, doge2
    dataset_path: str
    output_dir: str
    model_config: str   # Path to model config file
    additional_args: Optional[Dict[str, Any]] = None


####################
# Training Discovery Endpoints
####################

@router.get("/types")
async def get_training_types() -> Dict[str, Any]:
    """
    Get available training types
    
    Returns:
        Dict: Available training types
    """
    try:
        types = get_available_training_types()
        return {
            "training_types": types,
            "count": len(types)
        }
    except Exception as e:
        logger.error(f"Error getting training types: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/models")
async def get_models() -> Dict[str, Any]:
    """
    Get available model architectures
    
    Returns:
        Dict: Available model types
    """
    try:
        models = get_available_models()
        return {
            "model_types": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Error getting model types: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/configs/{model_type}")
async def get_configs(model_type: str) -> Dict[str, Any]:
    """
    Get available training configurations for a model type
    
    Args:
        model_type: Model architecture (doge, doge2)
        
    Returns:
        Dict: Available configurations
    """
    try:
        configs = get_training_configs(model_type)
        return {
            "model_type": model_type,
            "configurations": configs,
            "count": len(configs)
        }
    except Exception as e:
        logger.error(f"Error getting configs for {model_type}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Training Management Endpoints
####################

@router.post("/start")
async def start_training_job(request: TrainingStartRequest) -> Dict[str, Any]:
    """
    Start a training job
    
    Args:
        request: Training start request parameters
        
    Returns:
        Dict: Training start result
    """
    try:
        result = start_training(
            training_type=request.training_type,
            model_type=request.model_type,
            dataset_path=request.dataset_path,
            output_dir=request.output_dir,
            model_config=request.model_config,
            additional_args=request.additional_args
        )
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/status/{job_name}")
async def get_job_status(
    job_name: str,
    base_output_dir: str = Query("./data/training", description="Base training output directory")
) -> Dict[str, Any]:
    """
    Get status of a training job
    
    Args:
        job_name: Name of the training job
        base_output_dir: Base directory for training outputs
        
    Returns:
        Dict: Training job status
    """
    try:
        import os
        output_dir = os.path.join(base_output_dir, job_name)
        status_info = get_training_status(output_dir)
        
        return {
            "job_name": job_name,
            "output_dir": output_dir,
            **status_info
        }
        
    except Exception as e:
        logger.error(f"Error getting training status for {job_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/jobs")
async def list_jobs(
    base_output_dir: str = Query("./data/training", description="Base training output directory")
) -> Dict[str, Any]:
    """
    List all training jobs
    
    Args:
        base_output_dir: Base directory for training outputs
        
    Returns:
        Dict: List of training jobs
    """
    try:
        jobs = list_training_jobs(base_output_dir)
        return {
            "jobs": jobs,
            "count": len(jobs),
            "base_output_dir": base_output_dir
        }
        
    except Exception as e:
        logger.error(f"Error listing training jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/logs/{job_name}")
async def get_job_logs(
    job_name: str,
    base_output_dir: str = Query("./data/training", description="Base training output directory")
) -> Dict[str, Any]:
    """
    Get logs for a training job
    
    Args:
        job_name: Name of the training job
        base_output_dir: Base directory for training outputs
        
    Returns:
        Dict: Training job logs
    """
    try:
        import os
        output_dir = os.path.join(base_output_dir, job_name)
        logs_info = get_training_logs(output_dir)
        
        return {
            "job_name": job_name,
            "output_dir": output_dir,
            **logs_info
        }
        
    except Exception as e:
        logger.error(f"Error getting logs for {job_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.post("/stop/{job_name}")
async def stop_training_job(
    job_name: str,
    base_output_dir: str = Query("./data/training", description="Base training output directory")
) -> Dict[str, Any]:
    """
    Stop a training job
    
    Args:
        job_name: Name of the training job to stop
        base_output_dir: Base directory for training outputs
        
    Returns:
        Dict: Stop result
    """
    try:
        import os
        output_dir = os.path.join(base_output_dir, job_name)
        result = stop_training(output_dir)
        
        return {
            "job_name": job_name,
            "output_dir": output_dir,
            **result
        }
        
    except Exception as e:
        logger.error(f"Error stopping training job {job_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Training Information Endpoints
####################

@router.get("/info")
async def get_training_info() -> Dict[str, Any]:
    """
    Get general training information
    
    Returns:
        Dict: Training system information
    """
    try:
        return {
            "training_types": get_available_training_types(),
            "model_types": get_available_models(),
            "description": "SmallDoge training system supports pretrain, SFT, DPO, and GRPO training",
            "supported_architectures": ["doge", "doge2"]
        }
        
    except Exception as e:
        logger.error(f"Error getting training info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )