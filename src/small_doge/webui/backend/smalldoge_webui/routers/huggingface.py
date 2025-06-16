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
HuggingFace Hub router for SmallDoge WebUI
Provides endpoints for model discovery, search, and integration
"""

import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel

from smalldoge_webui.utils.models import (
    search_models_by_task,
    get_available_task_categories,
    check_huggingface_model_compatibility,
    get_model_families,
    get_popular_models_by_category,
    validate_and_load_huggingface_model
)
from smalldoge_webui.utils.huggingface_integration import search_huggingface_models
from smalldoge_webui.constants import ERROR_MESSAGES

log = logging.getLogger(__name__)
router = APIRouter()


####################
# Request/Response Models
####################

class ModelSearchRequest(BaseModel):
    """Model search request"""
    task: Optional[str] = None
    query: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    limit: int = 20


class ModelCompatibilityRequest(BaseModel):
    """Model compatibility check request"""
    model_id: str


class ModelLoadRequest(BaseModel):
    """Model load request"""
    model_id: str
    force_reload: bool = False


####################
# Search Endpoints
####################

@router.get("/search")
async def search_huggingface_models_endpoint(
    task: Optional[str] = Query(None, description="Task category (e.g., text-generation)"),
    query: Optional[str] = Query(None, description="Search query string"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    category: Optional[str] = Query(None, description="Model category"),
    limit: int = Query(20, description="Maximum number of results")
) -> Dict[str, Any]:
    """
    Search HuggingFace models with various filters

    Returns:
        Dict containing search results and metadata
    """
    try:
        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]

        # Use comprehensive search that combines all criteria
        # Instead of prioritizing one over another, pass all parameters
        
        results_data = search_huggingface_models(
            task=task if task else None,
            query=query if query else None,
            tags=tag_list if tag_list else None,
            limit=limit
        )
        
        # If no results and we have specific search criteria, try fallback searches
        if not results_data and (query or task or tag_list):
            # Try query-only search if we have a query
            if query:
                log.info(f"Primary search failed, trying query-only search for: {query}")
                results_data = search_huggingface_models(query=query, limit=limit)
            
            # If still no results, try task-only search
            if not results_data and task:
                log.info(f"Query search failed, trying task-only search for: {task}")
                results_data = search_huggingface_models(task=task, limit=limit)
            
            # If still no results, try tag-only search
            if not results_data and tag_list:
                log.info(f"Task search failed, trying tag-only search for: {tag_list}")
                results_data = search_huggingface_models(tags=tag_list, limit=limit)

        # If still no results, get some default models
        if not results_data:
            log.info("All searches failed, returning default text-generation models")
            results_data = search_huggingface_models(task="text-generation", limit=limit)

        return {
            "results": results_data,
            "total": len(results_data),
            "search_params": {
                "task": task,
                "query": query,
                "tags": tag_list,
                "category": category,
                "limit": limit
            }
        }

    except Exception as e:
        log.error(f"Error searching HuggingFace models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.post("/search")
async def search_huggingface_models_post_endpoint(request: ModelSearchRequest) -> Dict[str, Any]:
    """
    Search HuggingFace models using POST request with body

    Args:
        request: Search request parameters

    Returns:
        Dict containing search results and metadata
    """
    try:
        # Use comprehensive search that combines all criteria
        
        results_data = search_huggingface_models(
            task=request.task if request.task else None,
            query=request.query if request.query else None,
            tags=request.tags if request.tags else None,
            limit=request.limit
        )
        
        # If no results and we have specific search criteria, try fallback searches
        if not results_data and (request.query or request.task or request.tags):
            # Try query-only search if we have a query
            if request.query:
                log.info(f"Primary search failed, trying query-only search for: {request.query}")
                results_data = search_huggingface_models(query=request.query, limit=request.limit)
            
            # If still no results, try task-only search
            if not results_data and request.task:
                log.info(f"Query search failed, trying task-only search for: {request.task}")
                results_data = search_huggingface_models(task=request.task, limit=request.limit)
            
            # If still no results, try tag-only search
            if not results_data and request.tags:
                log.info(f"Task search failed, trying tag-only search for: {request.tags}")
                results_data = search_huggingface_models(tags=request.tags, limit=request.limit)

        # If still no results, get some default models
        if not results_data:
            log.info("All searches failed, returning default text-generation models")
            results_data = search_huggingface_models(task="text-generation", limit=request.limit)

        return {
            "results": results_data,
            "total": len(results_data),
            "search_params": request.dict()
        }

    except Exception as e:
        log.error(f"Error searching HuggingFace models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Discovery Endpoints
####################

@router.get("/categories")
async def get_task_categories() -> Dict[str, Any]:
    """
    Get available task categories for model search

    Returns:
        Dict containing available task categories
    """
    try:
        categories = get_available_task_categories()
        return {
            "categories": categories,
            "total": len(categories)
        }

    except Exception as e:
        log.error(f"Error getting task categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/families")
async def get_model_families_endpoint() -> Dict[str, Any]:
    """
    Get organized model families for browsing

    Returns:
        Dict containing model families organized by category
    """
    try:
        families = get_model_families()
        return {
            "families": families,
            "total_families": len(families),
            "total_models": sum(len(models) for models in families.values())
        }

    except Exception as e:
        log.error(f"Error getting model families: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/popular/{category}")
async def get_popular_models(
    category: str,
    limit: int = Query(10, description="Maximum number of results")
) -> Dict[str, Any]:
    """
    Get popular models by category

    Args:
        category: Model category
        limit: Maximum number of results

    Returns:
        Dict containing popular models for the category
    """
    try:
        results = get_popular_models_by_category(category, limit)
        return {
            "category": category,
            "results": results,
            "total": len(results)
        }

    except Exception as e:
        log.error(f"Error getting popular models for category {category}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Model Management Endpoints
####################

@router.get("/check-compatibility/{model_id:path}")
async def check_model_compatibility(model_id: str) -> Dict[str, Any]:
    """
    Check if a HuggingFace model is compatible with our system

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Dict containing compatibility information
    """
    try:
        compatibility = check_huggingface_model_compatibility(model_id)
        return compatibility

    except Exception as e:
        log.error(f"Error checking compatibility for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.post("/check-compatibility")
async def check_model_compatibility_post(
    request: ModelCompatibilityRequest
) -> Dict[str, Any]:
    """
    Check model compatibility using POST request

    Args:
        request: Compatibility check request

    Returns:
        Dict containing compatibility information
    """
    try:
        compatibility = check_huggingface_model_compatibility(request.model_id)
        return compatibility

    except Exception as e:
        log.error(f"Error checking compatibility for {request.model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.post("/load-model")
async def load_huggingface_model(request: ModelLoadRequest) -> Dict[str, Any]:
    """
    Validate and load a HuggingFace model

    Args:
        request: Model load request

    Returns:
        Dict containing load results
    """
    try:
        result = await validate_and_load_huggingface_model(request.model_id)
        
        if not result["loaded"]:
            # If loading failed, return appropriate HTTP status
            if "not compatible" in result.get("error", "").lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model {request.model_id} is not compatible: {result['error']}"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load model {request.model_id}: {result.get('error', 'Unknown error')}"
                )
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error loading HuggingFace model {request.model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


####################
# Quick Actions
####################

@router.get("/trending")
async def get_trending_models(
    limit: int = Query(10, description="Maximum number of results")
) -> Dict[str, Any]:
    """
    Get trending (recently popular) models

    Args:
        limit: Maximum number of results

    Returns:
        Dict containing trending models
    """
    try:
        # Get recently popular models across different categories
        results = search_models_by_task("text-generation", limit)
        
        return {
            "results": results,
            "total": len(results),
            "description": "Recently popular models for text generation"
        }

    except Exception as e:
        log.error(f"Error getting trending models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.get("/featured")
async def get_featured_models() -> Dict[str, Any]:
    """
    Get featured models across different categories

    Returns:
        Dict containing featured models by category
    """
    try:
        featured = {}
        categories = ["chat", "code", "instruct", "small", "multilingual"]
        
        for category in categories:
            featured[category] = get_popular_models_by_category(category, 5)
        
        return {
            "featured": featured,
            "categories": list(featured.keys()),
            "total_models": sum(len(models) for models in featured.values())
        }

    except Exception as e:
        log.error(f"Error getting featured models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )
