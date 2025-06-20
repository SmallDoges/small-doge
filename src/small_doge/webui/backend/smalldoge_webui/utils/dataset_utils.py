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
Dataset utilities for SmallDoge WebUI
Wraps existing dataset download and processing functions for API access
"""

import os
import multiprocessing
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)


# Pretrain dataset download functions
def download_fineweb_edu(save_dir: str, cache_dir: str, num_proc: int = 1) -> Dict[str, Any]:
    """Download FineWeb-Edu dataset"""
    try:
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "fineweb-edu-dedup",
            split="train",
            num_proc=num_proc,
            cache_dir=cache_dir,
        )
        dataset.save_to_disk(os.path.join(save_dir, "fineweb-edu"), num_proc=num_proc)
        return {
            "status": "success",
            "message": f"FineWeb-Edu dataset downloaded to {save_dir}/fineweb-edu",
            "dataset_info": str(dataset)
        }
    except Exception as e:
        logger.error(f"FineWeb-Edu download error: {e}")
        return {
            "status": "error",
            "message": f"Failed to download FineWeb-Edu: {str(e)}"
        }


def download_cosmopedia_v2(save_dir: str, cache_dir: str, num_proc: int = 1) -> Dict[str, Any]:
    """Download Cosmopedia-v2 dataset"""
    try:
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "cosmopedia-v2",
            split="train",
            num_proc=num_proc,
            cache_dir=cache_dir,
        )
        dataset.save_to_disk(os.path.join(save_dir, "cosmopedia-v2"), num_proc=num_proc)
        return {
            "status": "success",
            "message": f"Cosmopedia-v2 dataset downloaded to {save_dir}/cosmopedia-v2",
            "dataset_info": str(dataset)
        }
    except Exception as e:
        logger.error(f"Cosmopedia-v2 download error: {e}")
        return {
            "status": "error",
            "message": f"Failed to download Cosmopedia-v2: {str(e)}"
        }


def download_fine_math(save_dir: str, cache_dir: str, num_proc: int = 1) -> Dict[str, Any]:
    """Download FineMath dataset"""
    try:
        dataset = load_dataset(
            "HuggingFaceTB/finemath",
            "finemath-4plus",
            split="train",
            num_proc=num_proc,
            cache_dir=cache_dir,
        )
        dataset.save_to_disk(os.path.join(save_dir, "finemath"), num_proc=num_proc)
        return {
            "status": "success",
            "message": f"FineMath dataset downloaded to {save_dir}/finemath",
            "dataset_info": str(dataset)
        }
    except Exception as e:
        logger.error(f"FineMath download error: {e}")
        return {
            "status": "error",
            "message": f"Failed to download FineMath: {str(e)}"
        }


# Fine-tuning dataset download functions
def download_smoltalk(save_dir: str, cache_dir: str, num_proc: int = 1) -> Dict[str, Any]:
    """Download SmolTalk dataset"""
    try:
        dataset = load_dataset("HuggingFaceTB/smoltalk", "all", num_proc=num_proc, cache_dir=cache_dir)
        dataset.save_to_disk(os.path.join(save_dir, "smoltalk"), num_proc=num_proc)
        return {
            "status": "success",
            "message": f"SmolTalk dataset downloaded to {save_dir}/smoltalk",
            "dataset_info": str(dataset)
        }
    except Exception as e:
        logger.error(f"SmolTalk download error: {e}")
        return {
            "status": "error",
            "message": f"Failed to download SmolTalk: {str(e)}"
        }


def download_ultrafeedback_binarized(save_dir: str, cache_dir: str, num_proc: int = 1) -> Dict[str, Any]:
    """Download UltraFeedback Binarized dataset"""
    try:
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", num_proc=num_proc, cache_dir=cache_dir)
        dataset.save_to_disk(os.path.join(save_dir, "ultrafeedback_binarized"), num_proc=num_proc)
        return {
            "status": "success",
            "message": f"UltraFeedback Binarized dataset downloaded to {save_dir}/ultrafeedback_binarized",
            "dataset_info": str(dataset)
        }
    except Exception as e:
        logger.error(f"UltraFeedback download error: {e}")
        return {
            "status": "error",
            "message": f"Failed to download UltraFeedback Binarized: {str(e)}"
        }


def download_open_thoughts(save_dir: str, cache_dir: str, num_proc: int = 1) -> Dict[str, Any]:
    """Download Open Thoughts dataset"""
    try:
        dataset = load_dataset("open-thoughts/OpenThoughts-114k", "default", num_proc=num_proc, cache_dir=cache_dir)
        dataset.save_to_disk(os.path.join(save_dir, "open_thoughts"), num_proc=num_proc)
        return {
            "status": "success", 
            "message": f"Open Thoughts dataset downloaded to {save_dir}/open_thoughts",
            "dataset_info": str(dataset)
        }
    except Exception as e:
        logger.error(f"Open Thoughts download error: {e}")
        return {
            "status": "error",
            "message": f"Failed to download Open Thoughts: {str(e)}"
        }


def get_available_datasets() -> Dict[str, List[str]]:
    """Get list of available datasets for download"""
    return {
        "pretrain": [
            "fineweb-edu",
            "cosmopedia-v2", 
            "finemath"
        ],
        "finetune": [
            "smoltalk",
            "ultrafeedback_binarized",
            "open_thoughts"
        ]
    }


def get_downloaded_datasets(datasets_dir: str) -> List[str]:
    """Get list of already downloaded datasets"""
    if not os.path.exists(datasets_dir):
        return []
    
    downloaded = []
    for item in os.listdir(datasets_dir):
        item_path = os.path.join(datasets_dir, item)
        if os.path.isdir(item_path):
            downloaded.append(item)
    
    return downloaded


def download_dataset(dataset_name: str, save_dir: str, cache_dir: str, num_proc: int = 1) -> Dict[str, Any]:
    """Download a specific dataset by name"""
    download_functions = {
        "fineweb-edu": download_fineweb_edu,
        "cosmopedia-v2": download_cosmopedia_v2,
        "finemath": download_fine_math,
        "smoltalk": download_smoltalk,
        "ultrafeedback_binarized": download_ultrafeedback_binarized,
        "open_thoughts": download_open_thoughts,
    }
    
    if dataset_name not in download_functions:
        return {
            "status": "error",
            "message": f"Unknown dataset: {dataset_name}. Available datasets: {list(download_functions.keys())}"
        }
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    return download_functions[dataset_name](save_dir, cache_dir, num_proc)


def download_multiple_datasets(
    dataset_names: List[str], 
    save_dir: str, 
    cache_dir: str, 
    num_proc: int = 1,
    parallel: bool = False
) -> Dict[str, Any]:
    """Download multiple datasets"""
    if parallel:
        # TODO: Implement parallel download using multiprocessing
        # For now, download sequentially to avoid complexity
        pass
    
    results = {}
    for dataset_name in dataset_names:
        results[dataset_name] = download_dataset(dataset_name, save_dir, cache_dir, num_proc)
    
    return {
        "status": "completed",
        "results": results,
        "message": f"Downloaded {len(dataset_names)} datasets"
    }