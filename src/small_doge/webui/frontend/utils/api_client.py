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
Enhanced API client for SmallDoge WebUI frontend
Provides better error handling and streaming support
"""

import requests
import requests.exceptions
import json
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, AsyncGenerator, Generator
import logging

log = logging.getLogger(__name__)


class SmallDogeAPIClient:
    """Enhanced API client with streaming support and HuggingFace integration"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/openai"
        self.models_api = f"{base_url}/api/v1/models"
        self.hf_api = f"{base_url}/api/v1/huggingface"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def get_models(self) -> List[str]:
        """Get available models"""
        try:
            response = self.session.get(f"{self.api_base}/models")
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data["data"]]
        except Exception as e:
            log.error(f"Error getting models: {e}")
            return ["SmallDoge/Doge-160M"]  # Fallback
    
    def chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        model: str = "SmallDoge/Doge-160M",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Synchronous chat completion"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": stream
            }
            
            response = self.session.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._process_streaming_response(response)
            else:
                return response.json()
                
        except Exception as e:
            log.error(f"Chat completion error: {e}")
            raise
    
    def chat_completion_streaming(
        self,
        messages: List[Dict[str, str]],
        model: str = "SmallDoge/Doge-160M",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9
    ) -> Generator[str, None, None]:
        """Streaming chat completion generator"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": True
            }
            
            response = self.session.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and data['choices']:
                                choice = data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    yield choice['delta']['content']
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            log.error(f"Streaming chat completion error: {e}")
            raise
    
    def _process_streaming_response(self, response) -> Dict[str, Any]:
        """Process streaming response for non-generator usage"""
        content = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and data['choices']:
                            choice = data['choices'][0]
                            if 'delta' in choice and 'content' in choice['delta']:
                                content += choice['delta']['content']
                    except json.JSONDecodeError:
                        continue
        
        # Return in OpenAI format
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }]
        }
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/models/{model_id}/info")
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                log.warning(f"Model info endpoint not found for {model_id}")
                return {"id": model_id, "status": "info_unavailable"}
            else:
                response.raise_for_status()
                return response.json()
        except Exception as e:
            log.error(f"Error getting model info for {model_id}: {e}")
            return {"id": model_id, "status": "error", "error": str(e)}
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get model status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/models/{model_id}/status")
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                log.warning(f"Model status endpoint not found for {model_id}")
                return {"id": model_id, "status": "status_unavailable"}
            else:
                response.raise_for_status()
                return response.json()
        except Exception as e:
            log.error(f"Error getting model status for {model_id}: {e}")
            return {"id": model_id, "status": "error", "error": str(e)}
    
    def health_check(self) -> bool:
        """Check if backend is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    # HuggingFace Integration Methods
    def search_huggingface_models(
        self,
        task: str = None,
        query: str = None,
        tags: List[str] = None,
        category: str = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Search HuggingFace models"""
        try:
            params = {"limit": limit}
            if task:
                params["task"] = task
            if query:
                params["query"] = query
            if tags:
                params["tags"] = ",".join(tags)
            if category:
                params["category"] = category
            
            print(f"🔍 Making API request to: {self.hf_api}/search")
            print(f"🔍 Request params: {params}")
            
            response = self.session.get(f"{self.hf_api}/search", params=params)
            print(f"🔍 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"🔍 API response keys: {list(result.keys())}")
                return result
            else:
                print(f"❌ API error: {response.status_code} - {response.text}")
                response.raise_for_status()
                
        except requests.exceptions.ConnectionError as e:
            log.error(f"Connection error searching HuggingFace models: {e}")
            print(f"❌ Connection error: Backend may not be running")
            return {
                "results": self._get_fallback_models(task, query, tags),
                "total": 3,
                "error": "Backend connection failed - showing fallback models"
            }
        except requests.exceptions.Timeout as e:
            log.error(f"Timeout searching HuggingFace models: {e}")
            return {
                "results": self._get_fallback_models(task, query, tags),
                "total": 3,
                "error": "Request timeout - showing fallback models"
            }
        except Exception as e:
            log.error(f"Error searching HuggingFace models: {e}")
            print(f"❌ Unexpected error: {type(e).__name__}: {e}")
            return {
                "results": self._get_fallback_models(task, query, tags),
                "total": 3,
                "error": f"Search failed: {str(e)}"
            }
    
    def _get_fallback_models(self, task: str = None, query: str = None, tags: List[str] = None) -> List[Dict[str, Any]]:
        """Get fallback models when API is unavailable"""
        fallback_models = [
            {
                "model_id": "SmallDoge/Doge-160M",
                "task": "text-generation",
                "pipeline_tag": "text-generation",
                "tags": ["text-generation", "causal-lm", "small-model"],
                "downloads": 1000,
                "likes": 50,
                "description": "SmallDoge 160M parameter model for text generation",
                "compatible": True
            },
            {
                "model_id": "microsoft/DialoGPT-small",
                "task": "conversational",
                "pipeline_tag": "conversational",
                "tags": ["conversational", "chat", "dialogue"],
                "downloads": 500000,
                "likes": 200,
                "description": "Small conversational AI model",
                "compatible": True
            },
            {
                "model_id": "distilbert-base-uncased-distilled-squad",
                "task": "question-answering",
                "pipeline_tag": "question-answering",
                "tags": ["question-answering", "bert", "distilled"],
                "downloads": 800000,
                "likes": 150,
                "description": "DistilBERT model for question answering",
                "compatible": True
            }
        ]
        
        # Filter based on search criteria
        if task:
            fallback_models = [m for m in fallback_models if m.get("task") == task or m.get("pipeline_tag") == task]
        
        if tags:
            filtered = []
            for model in fallback_models:
                model_tags = model.get("tags", [])
                if any(tag.lower() in [t.lower() for t in model_tags] for tag in tags):
                    filtered.append(model)
            fallback_models = filtered
        
        if query:
            query_lower = query.lower()
            fallback_models = [m for m in fallback_models if query_lower in m.get("model_id", "").lower() or query_lower in m.get("description", "").lower()]
        
        return fallback_models[:3]  # Return max 3 fallback models
    
    def get_task_categories(self) -> Dict[str, Any]:
        """Get available task categories"""
        try:
            response = self.session.get(f"{self.hf_api}/categories")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Error getting task categories: {e}")
            return {"categories": [], "total": 0, "error": str(e)}
    
    def get_model_families(self) -> Dict[str, Any]:
        """Get model families"""
        try:
            response = self.session.get(f"{self.hf_api}/families")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Error getting model families: {e}")
            return {"families": {}, "error": str(e)}
    
    def get_popular_models(self, category: str, limit: int = 10) -> Dict[str, Any]:
        """Get popular models by category"""
        try:
            response = self.session.get(f"{self.hf_api}/popular/{category}", params={"limit": limit})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Error getting popular models for {category}: {e}")
            return {"results": [], "total": 0, "error": str(e)}
    
    def check_model_compatibility(self, model_id: str) -> Dict[str, Any]:
        """Check if a model is compatible"""
        try:
            response = self.session.get(f"{self.hf_api}/check-compatibility/{model_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Error checking compatibility for {model_id}: {e}")
            return {"compatible": False, "error": str(e)}
    
    def load_huggingface_model(self, model_id: str) -> Dict[str, Any]:
        """Load a HuggingFace model"""
        try:
            response = requests.post(
                f"{self.hf_api}/load-model",
                json={"model_id": model_id},
                timeout=300  # 5 minutes timeout for model loading
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
    
    def remove_model(self, model_id: str) -> Dict[str, Any]:
        """Remove a model from available models list and unload it from memory"""
        try:
            response = requests.delete(
                f"{self.base_url}/models/{model_id}/remove",
                timeout=60
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": response.json().get("message", "Model removed successfully"),
                    "details": response.json().get("details", {})
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
    
    def get_featured_models(self) -> Dict[str, Any]:
        """Get featured models across categories"""
        try:
            response = self.session.get(f"{self.hf_api}/featured")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Error getting featured models: {e}")
            return {"featured": {}, "error": str(e)}
    
    def get_trending_models(self, limit: int = 10) -> Dict[str, Any]:
        """Get trending models"""
        try:
            response = self.session.get(f"{self.hf_api}/trending", params={"limit": limit})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Error getting trending models: {e}")
            return {"results": [], "total": 0, "error": str(e)}
    
    def close(self):
        """Close the session"""
        self.session.close()


class AsyncSmallDogeAPIClient:
    """Async version of the API client"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/openai"
    
    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        model: str = "SmallDoge/Doge-160M",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat completion"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                json=payload
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data_str = line[6:]
                        
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and data['choices']:
                                choice = data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    yield choice['delta']['content']
                        except json.JSONDecodeError:
                            continue
    
    async def get_models_async(self) -> List[str]:
        """Get available models asynchronously"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base}/models") as response:
                    data = await response.json()
                    return [model["id"] for model in data["data"]]
        except Exception as e:
            log.error(f"Error getting models: {e}")
            return ["SmallDoge/Doge-160M"]
