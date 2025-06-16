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
HuggingFace Hub API utilities for SmallDoge WebUI
Provides direct API calls to HuggingFace Hub for model search and discovery
"""

import logging
import requests
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

# HuggingFace API base URL
HF_API_BASE = "https://huggingface.co/api"

@dataclass
class HFModelInfo:
    """Simple model info structure from HF API"""
    id: str
    pipeline_tag: Optional[str]
    tags: List[str]
    downloads: int
    likes: int
    created_at: str
    last_modified: str
    library_name: Optional[str]
    private: bool
    gated: bool


class HuggingFaceAPI:
    """Direct HuggingFace Hub API client"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SmallDoge-WebUI/1.0'
        })
        self.cache = {}
        self.cache_duration = 1800  # 30 minutes
    
    def search_models_by_tags(
        self,
        tags: List[str],
        limit: int = 20,
        sort: str = "downloads",
        direction: str = "desc"
    ) -> List[HFModelInfo]:
        """
        Search models by tags
        
        Args:
            tags: List of tags to search for
            limit: Maximum number of results to return
            sort: Sort field (downloads, likes, created_at)
            direction: Sort direction (desc/asc)
            
        Returns:
            List of HFModelInfo objects
        """
        try:
            cache_key = f"tags_{','.join(sorted(tags))}_{limit}_{sort}_{direction}"
            
            # Check cache
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    log.info(f"Returning cached results for tags: {tags}")
                    return cached_data
            
            # Convert direction string to number for HF API
            direction_value = -1 if direction == "desc" else 1
            
            # Use a broader search and then filter
            params = {
                'limit': limit * 3,  # Get more results to filter
                'sort': sort,
                'direction': direction_value,
                'filter': 'transformers'
            }
            
            log.info(f"Searching HF models for tags: {tags}")
            
            response = self.session.get(f"{HF_API_BASE}/models", params=params)
            response.raise_for_status()
            
            models_data = response.json()
            
            # Filter by tags
            filtered_models = []
            for model_data in models_data:
                model_tags = model_data.get('tags', [])
                # Check if model contains any of the requested tags (case-insensitive)
                if any(any(tag.lower() in model_tag.lower() or model_tag.lower() in tag.lower() 
                          for model_tag in model_tags) for tag in tags):
                    filtered_models.append(model_data)
                    
                if len(filtered_models) >= limit:
                    break
            
            results = []
            for model_data in filtered_models[:limit]:
                try:
                    model_info = HFModelInfo(
                        id=model_data.get('id', ''),
                        pipeline_tag=model_data.get('pipeline_tag'),
                        tags=model_data.get('tags', []),
                        downloads=model_data.get('downloads', 0),
                        likes=model_data.get('likes', 0),
                        created_at=model_data.get('created_at', ''),
                        last_modified=model_data.get('last_modified', ''),
                        library_name=model_data.get('library_name'),
                        private=model_data.get('private', False),
                        gated=model_data.get('gated', False)
                    )
                    results.append(model_info)
                except Exception as e:
                    log.warning(f"Error processing model data: {e}")
                    continue
            
            # Cache results
            self.cache[cache_key] = (results, time.time())
            
            log.info(f"Found {len(results)} models for tags: {tags}")
            return results
            
        except Exception as e:
            log.error(f"Error searching models by tags {tags}: {e}")
            return []
    
    def search_models_by_task(
        self,
        task: str,
        limit: int = 20,
        sort: str = "downloads",
        direction: str = "desc"
    ) -> List[HFModelInfo]:
        """
        Search models by pipeline task
        
        Args:
            task: Pipeline task (e.g., "text-generation", "question-answering")
            limit: Maximum number of results to return
            sort: Sort field (downloads, likes, created_at)
            direction: Sort direction (desc/asc)
            
        Returns:
            List of HFModelInfo objects
        """
        try:
            cache_key = f"task_{task}_{limit}_{sort}_{direction}"
            
            # Check cache
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    log.info(f"Returning cached results for task: {task}")
                    return cached_data
            
            # Convert direction string to number for HF API
            direction_value = -1 if direction == "desc" else 1
            
            params = {
                'limit': limit,
                'sort': sort,
                'direction': direction_value,
                'filter': 'transformers'
            }
            
            # Only add pipeline_tag if it's a valid task
            if task and task.strip():
                params['pipeline_tag'] = task
            
            log.info(f"Searching HF models for task: {task}")
            
            response = self.session.get(f"{HF_API_BASE}/models", params=params)
            response.raise_for_status()
            
            models_data = response.json()
            
            results = []
            for model_data in models_data:
                try:
                    model_info = HFModelInfo(
                        id=model_data.get('id', ''),
                        pipeline_tag=model_data.get('pipeline_tag'),
                        tags=model_data.get('tags', []),
                        downloads=model_data.get('downloads', 0),
                        likes=model_data.get('likes', 0),
                        created_at=model_data.get('created_at', ''),
                        last_modified=model_data.get('last_modified', ''),
                        library_name=model_data.get('library_name'),
                        private=model_data.get('private', False),
                        gated=model_data.get('gated', False)
                    )
                    results.append(model_info)
                except Exception as e:
                    log.warning(f"Error processing model data: {e}")
                    continue
            
            # Cache results
            self.cache[cache_key] = (results, time.time())
            
            log.info(f"Found {len(results)} models for task: {task}")
            return results
            
        except Exception as e:
            log.error(f"Error searching models by task {task}: {e}")
            # Return some fallback models for text-generation if it fails
            if task == "text-generation":
                return [
                    HFModelInfo(
                        id="SmallDoge/Doge-160M",
                        pipeline_tag="text-generation",
                        tags=["text-generation", "causal-lm"],
                        downloads=1000,
                        likes=50,
                        created_at="2024-01-01",
                        last_modified="2024-01-01",
                        library_name="transformers",
                        private=False,
                        gated=False
                    )
                ]
            return []
    
    def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        task: Optional[str] = None,
        limit: int = 20,
        sort: str = "downloads",
        direction: str = "desc"
    ) -> List[HFModelInfo]:
        """
        General model search with multiple criteria
        
        Args:
            query: Search query string
            tags: List of tags to filter by
            task: Pipeline task to filter by
            limit: Maximum number of results
            sort: Sort field
            direction: Sort direction
            
        Returns:
            List of HFModelInfo objects
        """
        try:
            # Convert direction string to number for HF API
            direction_value = -1 if direction == "desc" else 1
            
            params = {
                'limit': limit if not tags else limit * 2,  # Get more if filtering by tags
                'sort': sort,
                'direction': direction_value,
                'filter': 'transformers'
            }
            
            if task and task.strip():
                params['pipeline_tag'] = task
            
            if query and query.strip():
                params['search'] = query
            
            log.info(f"Searching HF models - Query: {query}, Tags: {tags}, Task: {task}")
            
            response = self.session.get(f"{HF_API_BASE}/models", params=params)
            response.raise_for_status()
            
            models_data = response.json()
            
            # Filter by tags if specified
            if tags:
                filtered_models = []
                for model_data in models_data:
                    model_tags = model_data.get('tags', [])
                    # Check if model contains any of the requested tags
                    if any(any(tag.lower() in model_tag.lower() or model_tag.lower() in tag.lower() 
                              for model_tag in model_tags) for tag in tags):
                        filtered_models.append(model_data)
                models_data = filtered_models
            
            results = []
            for model_data in models_data[:limit]:
                try:
                    model_info = HFModelInfo(
                        id=model_data.get('id', ''),
                        pipeline_tag=model_data.get('pipeline_tag'),
                        tags=model_data.get('tags', []),
                        downloads=model_data.get('downloads', 0),
                        likes=model_data.get('likes', 0),
                        created_at=model_data.get('created_at', ''),
                        last_modified=model_data.get('last_modified', ''),
                        library_name=model_data.get('library_name'),
                        private=model_data.get('private', False),
                        gated=model_data.get('gated', False)
                    )
                    results.append(model_info)
                except Exception as e:
                    log.warning(f"Error processing model data: {e}")
                    continue
            
            log.info(f"Found {len(results)} models")
            return results
            
        except Exception as e:
            log.error(f"Error in general model search: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with model information or None if not found
        """
        try:
            log.info(f"Getting model info for: {model_id}")
            
            response = self.session.get(f"{HF_API_BASE}/models/{model_id}")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            log.error(f"Error getting model info for {model_id}: {e}")
            return None
    
    def check_model_compatibility(self, model_id: str) -> Dict[str, Any]:
        """
        Check if a model is compatible with SmallDoge WebUI
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with compatibility information
        """
        try:
            model_info = self.get_model_info(model_id)
            if not model_info:
                return {
                    "compatible": False,
                    "reason": "Model not found",
                    "model_id": model_id
                }
            
            compatibility = {
                "model_id": model_id,
                "compatible": True,
                "issues": [],
                "warnings": [],
                "requirements": []
            }
            
            # Check library compatibility
            library = model_info.get('library_name', 'transformers')
            if library not in ['transformers', 'pytorch']:
                compatibility["compatible"] = False
                compatibility["issues"].append(f"Unsupported library: {library}")
            
            # Check pipeline tag
            pipeline_tag = model_info.get('pipeline_tag', '')
            supported_tasks = [
                'text-generation', 'text2text-generation', 'conversational',
                'question-answering', 'text-classification', 'summarization'
            ]
            
            if pipeline_tag not in supported_tasks:
                compatibility["warnings"].append(f"Task '{pipeline_tag}' may not be fully supported")
            
            # Check for gated models
            if model_info.get('gated', False):
                compatibility["warnings"].append("Model requires authentication (gated)")
                compatibility["requirements"].append("HuggingFace authentication token")
            
            # Check model tags for compatibility indicators
            tags = model_info.get('tags', [])
            if 'safetensors' in tags:
                compatibility["requirements"].append("Model uses safetensors format (recommended)")
            elif 'pytorch' not in tags and 'tf' not in tags:
                compatibility["warnings"].append("Model format unclear - may require special handling")
            
            # Set overall compatibility
            if compatibility["issues"]:
                compatibility["compatible"] = False
                compatibility["reason"] = "; ".join(compatibility["issues"])
            else:
                compatibility["reason"] = "Model appears compatible"
            
            return compatibility
            
        except Exception as e:
            log.error(f"Error checking compatibility for {model_id}: {e}")
            return {
                "compatible": False,
                "reason": f"Error checking compatibility: {str(e)}",
                "model_id": model_id
            }


# Global API instance
hf_api = HuggingFaceAPI()


# Public interface functions
def search_by_tags(tags: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search models by tags and return simplified model information
    
    Args:
        tags: List of tags to search for
        limit: Maximum number of results
        
    Returns:
        List of dictionaries with model information
    """
    try:
        models = hf_api.search_models_by_tags(tags, limit)
        
        return [
            {
                "model_id": model.id,
                "pipeline_tag": model.pipeline_tag,
                "tags": model.tags,
                "downloads": model.downloads,
                "likes": model.likes,
                "compatible": _is_compatible(model),
                "description": f"HuggingFace model for {model.pipeline_tag or 'various'} tasks",
                "gated": model.gated,
                "private": model.private
            }
            for model in models
        ]
        
    except Exception as e:
        log.error(f"Error in search_by_tags: {e}")
        return []


def search_by_task(task: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search models by pipeline task
    
    Args:
        task: Pipeline task (e.g., "text-generation", "question-answering")
        limit: Maximum number of results
        
    Returns:
        List of dictionaries with model information
    """
    try:
        models = hf_api.search_models_by_task(task, limit)
        
        return [
            {
                "model_id": model.id,
                "pipeline_tag": model.pipeline_tag,
                "tags": model.tags,
                "downloads": model.downloads,
                "likes": model.likes,
                "compatible": _is_compatible(model),
                "description": f"HuggingFace model for {task} tasks",
                "gated": model.gated,
                "private": model.private
            }
            for model in models
        ]
        
    except Exception as e:
        log.error(f"Error in search_by_task: {e}")
        return []


def search_models_general(
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    task: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    General model search function
    
    Args:
        query: Search query string
        tags: List of tags to filter by
        task: Pipeline task to filter by
        limit: Maximum number of results
        
    Returns:
        List of dictionaries with model information
    """
    try:
        models = hf_api.search_models(query=query, tags=tags, task=task, limit=limit)
        
        return [
            {
                "model_id": model.id,
                "pipeline_tag": model.pipeline_tag,
                "tags": model.tags,
                "downloads": model.downloads,
                "likes": model.likes,
                "compatible": _is_compatible(model),
                "description": f"HuggingFace model for {model.pipeline_tag or 'various'} tasks",
                "gated": model.gated,
                "private": model.private
            }
            for model in models
        ]
        
    except Exception as e:
        log.error(f"Error in search_models_general: {e}")
        return []


def get_model_details(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific model
    
    Args:
        model_id: Model identifier
        
    Returns:
        Dictionary with detailed model information
    """
    return hf_api.get_model_info(model_id)


def check_compatibility(model_id: str) -> Dict[str, Any]:
    """
    Check model compatibility with SmallDoge WebUI
    
    Args:
        model_id: Model identifier
        
    Returns:
        Dictionary with compatibility information
    """
    return hf_api.check_model_compatibility(model_id)


def _is_compatible(model: HFModelInfo) -> bool:
    """
    Quick compatibility check for a model
    
    Args:
        model: HFModelInfo object
        
    Returns:
        True if model appears compatible
    """
    # Check library
    if model.library_name and model.library_name not in ['transformers', 'pytorch']:
        return False
    
    # Check if it's gated or private
    if model.gated or model.private:
        return False
    
    # Check pipeline tag
    supported_tasks = [
        'text-generation', 'text2text-generation', 'conversational',
        'question-answering', 'text-classification', 'summarization'
    ]
    
    if model.pipeline_tag and model.pipeline_tag not in supported_tasks:
        return False
    
    return True


# Common task categories for easy access
TASK_CATEGORIES = [
    {"value": "text-generation", "label": "Text Generation"},
    {"value": "question-answering", "label": "Question Answering"},
    {"value": "text-classification", "label": "Text Classification"},
    {"value": "conversational", "label": "Conversational AI"},
    {"value": "summarization", "label": "Summarization"},
    {"value": "translation", "label": "Translation"},
    {"value": "text2text-generation", "label": "Text-to-Text Generation"},
    {"value": "fill-mask", "label": "Fill Mask"},
    {"value": "token-classification", "label": "Token Classification"},
]


# Common tags for popular model types
POPULAR_TAGS = {
    "chat": ["conversational", "chat", "dialogue", "assistant"],
    "instruct": ["instruct", "instruction", "tuned", "finetuned"],
    "code": ["code", "programming", "codegen", "python"],
    "math": ["math", "mathematics", "reasoning", "logic"],
    "small": ["small", "lightweight", "mobile", "efficient"],
    "multilingual": ["multilingual", "translation", "cross-lingual"]
}
