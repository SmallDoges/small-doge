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
Hugging Face Hub Integration for SmallDoge WebUI
Provides intelligent model discovery, task-based search, and dynamic model loading
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

import requests
from huggingface_hub import HfApi, model_info
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from small_doge.webui.backend.smalldoge_webui.env import MODEL_CACHE_DIR
from small_doge.webui.backend.smalldoge_webui.constants import ERROR_MESSAGES
from small_doge.webui.backend.smalldoge_webui.utils.hf_api import hf_api, search_by_tags, search_by_task, search_models_general

log = logging.getLogger(__name__)


####################
# Task Categories and Tags
####################

class TaskCategory(Enum):
    """Supported task categories for model search"""
    TEXT_GENERATION = "text-generation"
    QUESTION_ANSWERING = "question-answering"
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CONVERSATIONAL = "conversational"
    TEXT2TEXT_GENERATION = "text2text-generation"
    FEATURE_EXTRACTION = "feature-extraction"
    FILL_MASK = "fill-mask"


@dataclass
class ModelMetadata:
    """Metadata for a Hugging Face model"""
    model_id: str
    task: str
    tags: List[str]
    downloads: int
    likes: int
    created_at: str
    updated_at: str
    library: str
    language: Optional[List[str]]
    license: Optional[str]
    model_size: Optional[str]
    pipeline_tag: Optional[str]
    compatible: bool
    description: str


class HuggingFaceIntegration:
    """Main class for Hugging Face Hub integration"""
    
    def __init__(self):
        self.hf_api = HfApi()
        self.supported_libraries = {"transformers", "pytorch"}
        self.supported_tasks = {task.value for task in TaskCategory}
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
    def search_models(
        self,
        task: Optional[str] = None,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
        sort_by: str = "downloads",
        limit: int = 20,
        filter_compatible: bool = True
    ) -> List[ModelMetadata]:
        """
        Search for models on Hugging Face Hub with intelligent filtering
        
        Args:
            task: Task category (e.g., "text-generation")
            query: Search query string
            tags: List of tags to filter by
            language: Language filter
            sort_by: Sort criteria ("downloads", "likes", "created_at")
            limit: Maximum number of results
            filter_compatible: Only return compatible models
            
        Returns:
            List[ModelMetadata]: List of model metadata
        """
        try:
            cache_key = f"{task}_{query}_{tags}_{language}_{sort_by}_{limit}"
            
            # Check cache
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    log.info(f"Returning cached results for search: {cache_key}")
                    return cached_data
            
            log.info(f"Searching HuggingFace models - Task: {task}, Query: {query}, Tags: {tags}")
            
            # Use the new hf_api for searching
            if tags and task:
                # Search by both tags and task
                models_data = search_models_general(query=query, tags=tags, task=task, limit=limit * 2)
            elif tags:
                # Search by tags only
                models_data = search_by_tags(tags, limit=limit * 2)
            elif task:
                # Search by task only
                models_data = search_by_task(task, limit=limit * 2)
            else:
                # General search
                models_data = search_models_general(query=query, limit=limit * 2)
            
            results = []
            for model_data in models_data:
                try:
                    # Convert API response to ModelMetadata
                    metadata = ModelMetadata(
                        model_id=model_data.get("model_id", ""),
                        task=model_data.get("pipeline_tag", "unknown"),
                        tags=model_data.get("tags", []),
                        downloads=model_data.get("downloads", 0),
                        likes=model_data.get("likes", 0),
                        created_at="",  # Will be filled if needed
                        updated_at="",  # Will be filled if needed
                        library="transformers",  # Default assumption
                        language=None,  # Language filter not implemented yet
                        license=None,  # License info not in basic search
                        model_size=self._extract_model_size_from_name(model_data.get("model_id", "")),
                        pipeline_tag=model_data.get("pipeline_tag"),
                        compatible=model_data.get("compatible", True),
                        description=model_data.get("description", "")
                    )
                    
                    # Filter compatible models if requested
                    if filter_compatible and not metadata.compatible:
                        continue
                    
                    results.append(metadata)
                    
                    if len(results) >= limit:
                        break
                        
                except Exception as e:
                    log.warning(f"Error processing model {model_data.get('model_id', 'unknown')}: {e}")
                    continue
            
            # Cache results
            self.cache[cache_key] = (results, time.time())
            
            log.info(f"Found {len(results)} compatible models")
            return results
            
        except Exception as e:
            log.error(f"Error searching models: {e}")
            return []
    
    def _extract_model_size_from_name(self, model_id: str) -> Optional[str]:
        """Extract model size from model name/ID"""
        try:
            model_name = model_id.lower()
            if '160m' in model_name:
                return '160M'
            elif '1b' in model_name:
                return '1B'
            elif '7b' in model_name:
                return '7B'
            elif '13b' in model_name:
                return '13B'
            elif '70b' in model_name:
                return '70B'
            return None
        except Exception:
            return None
    
    def _extract_model_metadata(self, model) -> ModelMetadata:
        """Extract metadata from a HuggingFace model"""
        try:
            # Get model info for additional details
            info = None
            try:
                info = model_info(model.modelId)
            except Exception as e:
                log.warning(f"Could not get detailed info for {model.modelId}: {e}")
            
            # Determine compatibility
            compatible = self._check_compatibility(model)
            
            # Extract model size if available
            model_size = self._extract_model_size(model, info)
            
            # Extract description
            description = ""
            if hasattr(model, 'card_data') and model.card_data:
                description = getattr(model.card_data, 'description', "")
            if not description and info and hasattr(info, 'card_data') and info.card_data:
                description = getattr(info.card_data, 'description', "")
            
            return ModelMetadata(
                model_id=model.modelId,
                task=getattr(model, 'pipeline_tag', 'unknown'),
                tags=list(model.tags) if model.tags else [],
                downloads=getattr(model, 'downloads', 0),
                likes=getattr(model, 'likes', 0),
                created_at=model.created_at.isoformat() if model.created_at else "",
                updated_at=model.last_modified.isoformat() if model.last_modified else "",
                library=getattr(model, 'library_name', 'transformers'),
                language=getattr(model, 'language', None),
                license=getattr(model, 'license', None),
                model_size=model_size,
                pipeline_tag=getattr(model, 'pipeline_tag', None),
                compatible=compatible,
                description=description or f"Hugging Face model for {getattr(model, 'pipeline_tag', 'various')} tasks"
            )
            
        except Exception as e:
            log.error(f"Error extracting metadata for {model.modelId}: {e}")
            raise
    
    def _check_compatibility(self, model) -> bool:
        """Check if a model is compatible with our system"""
        try:
            # Check library compatibility
            library = getattr(model, 'library_name', 'transformers')
            if library not in self.supported_libraries:
                return False
            
            # Check if it's a text generation or conversational model
            pipeline_tag = getattr(model, 'pipeline_tag', '')
            if pipeline_tag not in self.supported_tasks:
                return False
            
            # Check for problematic tags
            tags = list(model.tags) if model.tags else []
            problematic_tags = {'safetensors', 'gguf', 'onnx'}
            
            # If it has safetensors, that's good
            if 'safetensors' in tags:
                return True
            
            # If it only has problematic formats without pytorch, skip
            if any(tag in problematic_tags for tag in tags) and 'pytorch' not in tags:
                return False
            
            return True
            
        except Exception as e:
            log.warning(f"Error checking compatibility for {model.modelId}: {e}")
            return False
    
    def _extract_model_size(self, model, info) -> Optional[str]:
        """Extract model size information"""
        try:
            # Try to get from tags
            tags = list(model.tags) if model.tags else []
            size_tags = [tag for tag in tags if any(size in tag.lower() for size in ['7b', '13b', '70b', '160m', '1b', '3b'])]
            if size_tags:
                return size_tags[0]
            
            # Try to get from model name
            model_name = model.modelId.lower()
            if '160m' in model_name:
                return '160M'
            elif '1b' in model_name:
                return '1B'
            elif '7b' in model_name:
                return '7B'
            elif '13b' in model_name:
                return '13B'
            elif '70b' in model_name:
                return '70B'
            
            return None
            
        except Exception:
            return None
    
    def get_model_config_preview(self, model_id: str) -> Dict[str, Any]:
        """Get a preview of model configuration without downloading"""
        try:
            config = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=MODEL_CACHE_DIR
            )
            
            return {
                "model_type": getattr(config, 'model_type', 'unknown'),
                "vocab_size": getattr(config, 'vocab_size', 'unknown'),
                "hidden_size": getattr(config, 'hidden_size', 'unknown'),
                "num_attention_heads": getattr(config, 'num_attention_heads', 'unknown'),
                "num_hidden_layers": getattr(config, 'num_hidden_layers', 'unknown'),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', 'unknown'),
                "architectures": getattr(config, 'architectures', []),
            }
            
        except Exception as e:
            log.error(f"Error getting config preview for {model_id}: {e}")
            return {"error": str(e)}
    
    def check_model_compatibility(self, model_id: str) -> Dict[str, Any]:
        """Check detailed compatibility of a specific model"""
        try:
            # Get model info
            info = model_info(model_id)
            
            compatibility = {
                "model_id": model_id,
                "compatible": True,
                "issues": [],
                "warnings": [],
                "recommendations": []
            }
            
            # Check if config is accessible
            try:
                config = AutoConfig.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    cache_dir=MODEL_CACHE_DIR
                )
                compatibility["config_accessible"] = True
                compatibility["model_type"] = getattr(config, 'model_type', 'unknown')
            except Exception as e:
                compatibility["compatible"] = False
                compatibility["issues"].append(f"Config not accessible: {str(e)}")
                compatibility["config_accessible"] = False
            
            # Check if tokenizer is accessible
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    cache_dir=MODEL_CACHE_DIR
                )
                compatibility["tokenizer_accessible"] = True
            except Exception as e:
                compatibility["warnings"].append(f"Tokenizer issue: {str(e)}")
                compatibility["tokenizer_accessible"] = False
            
            # Check library and task compatibility
            if hasattr(info, 'library_name'):
                if info.library_name not in self.supported_libraries:
                    compatibility["compatible"] = False
                    compatibility["issues"].append(f"Unsupported library: {info.library_name}")
            
            if hasattr(info, 'pipeline_tag'):
                if info.pipeline_tag not in self.supported_tasks:
                    compatibility["warnings"].append(f"Task {info.pipeline_tag} may not be fully supported")
            
            # Add recommendations
            if compatibility["compatible"]:
                compatibility["recommendations"].append("Model appears compatible and ready to load")
            else:
                compatibility["recommendations"].append("Model may require additional configuration or is not supported")
            
            return compatibility
            
        except Exception as e:
            return {
                "model_id": model_id,
                "compatible": False,
                "error": str(e),
                "issues": [f"Failed to check compatibility: {str(e)}"]
            }
    
    def get_popular_models_by_task(self, task: str, limit: int = 10) -> List[ModelMetadata]:
        """Get popular models for a specific task"""
        return self.search_models(
            task=task,
            sort_by="downloads",
            limit=limit,
            filter_compatible=True
        )
    
    def get_trending_models(self, limit: int = 10) -> List[ModelMetadata]:
        """Get trending models (recently popular)"""
        return self.search_models(
            sort_by="created_at",
            limit=limit,
            filter_compatible=True
        )
    
    def search_by_capability(self, capability: str, limit: int = 10) -> List[ModelMetadata]:
        """Search models by specific capability"""
        capability_mapping = {
            "chat": ["conversational", "chat", "dialogue"],
            "code": ["code", "programming", "codegen"],
            "math": ["math", "mathematics", "reasoning"],
            "multilingual": ["multilingual", "translation"],
            "instruct": ["instruct", "instruction", "tuned"]
        }
        
        tags = capability_mapping.get(capability.lower(), [capability])
        
        return self.search_models(
            tags=tags,
            sort_by="downloads",
            limit=limit,
            filter_compatible=True
        )
    
    def get_model_families(self) -> Dict[str, List[str]]:
        """Get organized model families"""
        families = {
            "SmallDoge": ["SmallDoge/Doge-160M", "SmallDoge/Doge-1B", "SmallDoge/Doge-7B"],
            "Popular Small Models": [],
            "Code Models": [],
            "Chat Models": [],
            "Instruction Models": []
        }
        
        # Get popular small models
        small_models = self.search_models(
            query="small",
            sort_by="downloads",
            limit=5,
            filter_compatible=True
        )
        families["Popular Small Models"] = [m.model_id for m in small_models]
        
        # Get code models
        code_models = self.search_by_capability("code", limit=5)
        families["Code Models"] = [m.model_id for m in code_models]
        
        # Get chat models
        chat_models = self.search_by_capability("chat", limit=5)
        families["Chat Models"] = [m.model_id for m in chat_models]
        
        # Get instruction models
        instruct_models = self.search_by_capability("instruct", limit=5)
        families["Instruction Models"] = [m.model_id for m in instruct_models]
        
        return families

    def get_model_detailed_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed model information using HF API"""
        try:
            # Use the HF API to get detailed model info
            model_info_obj = self.hf_api.model_info(model_id)
            
            # Extract comprehensive information
            detailed_info = {
                "model_id": model_id,
                "author": model_info_obj.author,
                "created_at": model_info_obj.created_at.isoformat() if model_info_obj.created_at else None,
                "last_modified": model_info_obj.last_modified.isoformat() if model_info_obj.last_modified else None,
                "downloads": model_info_obj.downloads or 0,
                "downloads_all_time": model_info_obj.downloads_all_time or 0,
                "likes": model_info_obj.likes or 0,
                "tags": model_info_obj.tags or [],
                "pipeline_tag": model_info_obj.pipeline_tag,
                "library_name": model_info_obj.library_name,
                "private": model_info_obj.private,
                "gated": model_info_obj.gated,
                "disabled": model_info_obj.disabled,
                "safetensors": model_info_obj.safetensors,
                "transformers_info": model_info_obj.transformers_info,
                "config": model_info_obj.config,
                "model_index": model_info_obj.model_index,
                "card_data": model_info_obj.card_data,
                "siblings": [{"filename": s.rfilename, "size": s.size} for s in (model_info_obj.siblings or [])],
                "spaces": model_info_obj.spaces or [],
                "security_repo_status": model_info_obj.security_repo_status
            }
            
            return detailed_info
            
        except Exception as e:
            log.error(f"Error getting detailed info for {model_id}: {e}")
            return {"model_id": model_id, "error": str(e)}

    def get_model_files(self, model_id: str) -> List[Dict[str, Any]]:
        """Get list of files in a model repository"""
        try:
            repo_files = self.hf_api.list_repo_files(model_id, repo_type="model")
            
            files_info = []
            for file_path in repo_files:
                try:
                    # Get file metadata if possible
                    file_info = {
                        "filename": file_path,
                        "path": file_path,
                        "type": self._get_file_type(file_path)
                    }
                    files_info.append(file_info)
                except Exception as e:
                    log.warning(f"Could not get info for file {file_path}: {e}")
                    
            return files_info
            
        except Exception as e:
            log.error(f"Error getting files for {model_id}: {e}")
            return []

    def _get_file_type(self, filename: str) -> str:
        """Determine file type based on extension"""
        if filename.endswith('.safetensors'):
            return 'safetensors'
        elif filename.endswith('.bin'):
            return 'pytorch'
        elif filename.endswith('.json'):
            return 'config'
        elif filename.endswith('.txt'):
            return 'text'
        elif filename.endswith('.md'):
            return 'markdown'
        elif filename.endswith('.py'):
            return 'python'
        else:
            return 'other'    

    def search_models_advanced(
        self,
        task: Optional[str] = None,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
        library: Optional[str] = None,
        model_name: Optional[str] = None,
        author: Optional[str] = None,
        sort: str = "downloads",
        direction: int = -1,
        limit: int = 20,
        full: bool = False
    ) -> List[ModelMetadata]:
        """Advanced model search with more filtering options"""
        try:
            # Use the new hf_api for advanced search
            # For now, we'll use the general search and filter results
            models_data = search_models_general(
                query=query,
                tags=tags,
                task=task,
                limit=limit * 2
            )
            
            results = []
            for model_data in models_data:
                try:
                    # Additional filtering based on advanced parameters
                    model_id = model_data.get("model_id", "")
                    
                    # Filter by author if specified
                    if author and author.lower() not in model_id.lower():
                        continue
                    
                    # Filter by model name if specified
                    if model_name and model_name.lower() not in model_id.lower():
                        continue
                    
                    # Convert to ModelMetadata
                    metadata = ModelMetadata(
                        model_id=model_id,
                        task=model_data.get("pipeline_tag", "unknown"),
                        tags=model_data.get("tags", []),
                        downloads=model_data.get("downloads", 0),
                        likes=model_data.get("likes", 0),
                        created_at="",
                        updated_at="",
                        library="transformers",
                        language=None,
                        license=None,
                        model_size=self._extract_model_size_from_name(model_id),
                        pipeline_tag=model_data.get("pipeline_tag"),
                        compatible=model_data.get("compatible", True),
                        description=model_data.get("description", "")
                    )
                    
                    results.append(metadata)
                    
                    if len(results) >= limit:
                        break
                        
                except Exception as e:
                    log.warning(f"Error processing model {model_data.get('model_id', 'unknown')}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            log.error(f"Error in advanced search: {e}")
            return []

    def get_model_usage_examples(self, model_id: str) -> List[Dict[str, Any]]:
        """Get usage examples for a model"""
        try:
            model_info_obj = self.hf_api.model_info(model_id)
            
            examples = []
            
            # Check for widget data
            if model_info_obj.widget_data:
                for widget in model_info_obj.widget_data:
                    if isinstance(widget, dict) and 'text' in widget:
                        examples.append({
                            "type": "widget_example",
                            "text": widget['text'],
                            "parameters": widget.get('parameters', {})
                        })
            
            # Generate basic examples based on pipeline tag
            pipeline_tag = model_info_obj.pipeline_tag
            if pipeline_tag == "text-generation":
                examples.extend([
                    {"type": "basic", "text": "Once upon a time", "description": "Story generation"},
                    {"type": "basic", "text": "The future of AI is", "description": "Article writing"},
                ])
            elif pipeline_tag == "conversational":
                examples.extend([
                    {"type": "basic", "text": "Hello, how are you?", "description": "Casual conversation"},
                    {"type": "basic", "text": "Can you help me with", "description": "Assistant interaction"},
                ])
            elif pipeline_tag == "question-answering":
                examples.extend([
                    {"type": "basic", "text": "What is the capital of France?", "description": "Factual question"},
                    {"type": "basic", "text": "How does photosynthesis work?", "description": "Scientific question"},
                ])
            
            return examples[:5]  # Limit to 5 examples
            
        except Exception as e:
            log.error(f"Error getting usage examples for {model_id}: {e}")
            return []

    def get_similar_models(self, model_id: str, limit: int = 5) -> List[ModelMetadata]:
        """Find models similar to the given model"""
        try:
            # Get the target model info
            target_model = self.hf_api.model_info(model_id)
            
            # Search for similar models based on tags and task
            similar_models = []
            
            if target_model.tags:
                # Use the most relevant tags
                relevant_tags = [tag for tag in target_model.tags 
                               if tag in ['conversational', 'instruct', 'chat', 'code', 'math']][:3]
                
                if relevant_tags:
                    similar_models = self.search_models(
                        task=target_model.pipeline_tag,
                        tags=relevant_tags,
                        limit=limit + 5  # Get more to filter out the original
                    )
            
            # Filter out the original model and limit results
            similar_models = [m for m in similar_models if m.model_id != model_id][:limit]
            
            return similar_models
            
        except Exception as e:
            log.error(f"Error finding similar models for {model_id}: {e}")
            return []

    def get_model_performance_info(self, model_id: str) -> Dict[str, Any]:
        """Get performance information if available"""
        try:
            model_info_obj = self.hf_api.model_info(model_id)
            
            performance_info = {
                "model_id": model_id,
                "metrics": {},
                "benchmarks": [],
                "evaluation_results": []
            }
            
            # Check model-index for evaluation results
            if model_info_obj.model_index:
                for result in model_info_obj.model_index:
                    if 'results' in result:
                        for eval_result in result['results']:
                            performance_info["evaluation_results"].append({
                                "dataset": eval_result.get('dataset', {}).get('name', 'Unknown'),
                                "metrics": eval_result.get('metrics', {}),
                                "task": eval_result.get('task', {}).get('type', 'Unknown')
                            })
            
            # Extract metrics from card data if available
            if model_info_obj.card_data:
                card_data = model_info_obj.card_data
                if hasattr(card_data, 'model_index') and card_data.model_index:
                    for model_result in card_data.model_index:
                        if 'results' in model_result:
                            performance_info["benchmarks"].extend(model_result['results'])
            
            return performance_info
            
        except Exception as e:
            log.error(f"Error getting performance info for {model_id}: {e}")
            return {"model_id": model_id, "error": str(e)}


####################
# Global instance
####################

hf_integration = HuggingFaceIntegration()


####################
# Utility functions
####################

def search_huggingface_models(
    task: Optional[str] = None,
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Public function to search HuggingFace models"""
    try:
        results = hf_integration.search_models(
            task=task,
            query=query,
            tags=tags,
            limit=limit
        )
        
        return [
            {
                "model_id": r.model_id,
                "task": r.task,
                "tags": r.tags,
                "downloads": r.downloads,
                "likes": r.likes,
                "description": r.description,
                "model_size": r.model_size,
                "language": r.language,
                "license": r.license,
                "compatible": r.compatible
            }
            for r in results
        ]
        
    except Exception as e:
        log.error(f"Error in search_huggingface_models: {e}")
        return []


def get_task_categories() -> List[Dict[str, str]]:
    """Get available task categories"""
    return [
        {"value": task.value, "label": task.value.replace("-", " ").title()}
        for task in TaskCategory
    ]


def check_model_compatibility_public(model_id: str) -> Dict[str, Any]:
    """Public function to check model compatibility"""
    return hf_integration.check_model_compatibility(model_id)


def get_model_families_public() -> Dict[str, List[str]]:
    """Public function to get model families"""
    return hf_integration.get_model_families()
