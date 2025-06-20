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
Dataset management router for SmallDoge WebUI
Provides API endpoints for dataset download, processing, and management
"""

import logging
from typing import Dict, List, Any

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel

from smalldoge_webui.utils.dataset_utils import (
    get_available_datasets,
    get_downloaded_datasets,
    download_dataset,
    download_multiple_datasets
)
from smalldoge_webui.constants import ERROR_MESSAGES

logger = logging.getLogger(__name__)

router = APIRouter()


####################
# Request Models
####################

class DatasetDownloadRequest(BaseModel):
    dataset_name: str
    save_dir: str = "./data/datasets"
    cache_dir: str = "./data/cache"
    num_proc: int = 1


class MultipleDatasetDownloadRequest(BaseModel):
    dataset_names: List[str]
    save_dir: str = "./data/datasets"
    cache_dir: str = "./data/cache"
    num_proc: int = 1
    parallel: bool = False


####################
# Dataset Discovery Endpoints
####################

@router.get("/available")
async def get_available_datasets_endpoint() -> Dict[str, Any]:
    """
    Get list of available datasets for download
    
    Returns:
        Dict: Available datasets organized by type
    """
    try:
        datasets = get_available_datasets()
        return {
            "datasets": datasets,
            "total_count": sum(len(v) for v in datasets.values())
        }
    except Exception as e:
        logger.error(f"Error getting available datasets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/downloaded")
async def get_downloaded_datasets_endpoint(
    datasets_dir: str = Query("./data/datasets", description="Directory containing downloaded datasets")
) -> Dict[str, Any]:
    """
    Get list of already downloaded datasets
    
    Args:
        datasets_dir: Directory to check for downloaded datasets
        
    Returns:
        Dict: List of downloaded datasets
    """
    try:
        downloaded = get_downloaded_datasets(datasets_dir)
        return {
            "downloaded_datasets": downloaded,
            "count": len(downloaded),
            "datasets_dir": datasets_dir
        }
    except Exception as e:
        logger.error(f"Error getting downloaded datasets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Dataset Download Endpoints
####################

@router.post("/download")
async def download_dataset_endpoint(request: DatasetDownloadRequest) -> Dict[str, Any]:
    """
    Download a specific dataset
    
    Args:
        request: Dataset download request parameters
        
    Returns:
        Dict: Download result
    """
    try:
        result = download_dataset(
            dataset_name=request.dataset_name,
            save_dir=request.save_dir,
            cache_dir=request.cache_dir,
            num_proc=request.num_proc
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
        logger.error(f"Error downloading dataset {request.dataset_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.post("/download-multiple")
async def download_multiple_datasets_endpoint(request: MultipleDatasetDownloadRequest) -> Dict[str, Any]:
    """
    Download multiple datasets
    
    Args:
        request: Multiple dataset download request parameters
        
    Returns:
        Dict: Download results for all datasets
    """
    try:
        result = download_multiple_datasets(
            dataset_names=request.dataset_names,
            save_dir=request.save_dir,
            cache_dir=request.cache_dir,
            num_proc=request.num_proc,
            parallel=request.parallel
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error downloading multiple datasets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Dataset Information Endpoints
####################

@router.get("/info/{dataset_name}")
async def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a specific dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dict: Dataset information
    """
    try:
        available_datasets = get_available_datasets()
        all_datasets = []
        for dataset_list in available_datasets.values():
            all_datasets.extend(dataset_list)
        
        if dataset_name not in all_datasets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_name} not found"
            )
        
        # Basic dataset information
        dataset_info = {
            "name": dataset_name,
            "available": True,
            "description": f"Information about {dataset_name} dataset"
        }
        
        # Determine dataset type
        for dataset_type, datasets in available_datasets.items():
            if dataset_name in datasets:
                dataset_info["type"] = dataset_type
                break
        
        return dataset_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset info for {dataset_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Dataset Management Endpoints
####################

@router.delete("/delete/{dataset_name}")
async def delete_dataset(
    dataset_name: str,
    datasets_dir: str = Query("./data/datasets", description="Directory containing datasets")
) -> Dict[str, Any]:
    """
    Delete a downloaded dataset
    
    Args:
        dataset_name: Name of the dataset to delete
        datasets_dir: Directory containing datasets
        
    Returns:
        Dict: Deletion result
    """
    try:
        import shutil
        import os
        
        dataset_path = os.path.join(datasets_dir, dataset_name)
        
        if not os.path.exists(dataset_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_name} not found in {datasets_dir}"
            )
        
        shutil.rmtree(dataset_path)
        
        return {
            "status": "success",
            "message": f"Dataset {dataset_name} deleted successfully",
            "deleted_path": dataset_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )