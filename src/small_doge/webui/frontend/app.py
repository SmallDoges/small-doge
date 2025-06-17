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
Gradio Frontend for SmallDoge WebUI
Provides a chat interface that connects to the FastAPI backend
Enhanced with streaming support and improved UI/UX
"""

import gradio as gr
import requests
import json
import time
import asyncio
import aiohttp
import threading
from typing import List, Dict, Any, Optional, Generator
import os
from pathlib import Path
import uuid
from datetime import datetime

# Local imports
try:
    # Try relative import first (when run as module)
    from .utils.api_client import SmallDogeAPIClient
except ImportError:
    # Fallback to absolute import (when run directly)
    from utils.api_client import SmallDogeAPIClient

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/openai"
CHAT_HISTORY_FILE = Path("chat_history.json")


class SmallDogeWebUI:
    """Main Gradio interface for SmallDoge WebUI with Enhanced HuggingFace Integration"""

    def __init__(self):
        self.available_models = [
            "SmallDoge/Doge-320M-Instruct",
            "SmallDoge/Doge-160M-Instruct",
        ]  # Synchronized with backend MODEL_CONFIG.SMALLDOGE_MODELS
        self.huggingface_models = []
        self.chat_sessions = {}  # Store multiple chat sessions
        self.current_session_id = None
        self.api_client = SmallDogeAPIClient(BACKEND_URL)
        self.task_categories = []
        self.model_families = {}
        self.featured_models = {}
        self.search_results = []
        self.popular_tags = [
            "text-generation", "conversational", "chat", "instruction-following",
            "question-answering", "summarization", "translation", "code-generation",
            "small-model", "efficient", "fine-tuned", "multilingual"
        ]
        self.load_chat_history()
        self.load_initial_hf_data()

    # Authentication removed for open source sharing
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {"Content-Type": "application/json"}

    def load_chat_history(self):
        """Load chat history from file"""
        try:
            if CHAT_HISTORY_FILE.exists():
                with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chat_sessions = data.get('sessions', {})
                    self.current_session_id = data.get('current_session_id')
                    # Migrate old format to new format
                    self._migrate_chat_format()
        except Exception as e:
            print(f"Error loading chat history: {e}")
            self.chat_sessions = {}
            self.current_session_id = None

    def _migrate_chat_format(self):
        """Migrate old tuple-based format to new message format"""
        migrated = False
        for session_id, session_data in self.chat_sessions.items():
            messages = session_data.get('messages', [])
            if messages and isinstance(messages[0], list):
                # Old format: List[List[str]] -> convert to List[dict]
                new_messages = []
                for user_msg, assistant_msg in messages:
                    if user_msg:
                        new_messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        new_messages.append({"role": "assistant", "content": assistant_msg})
                session_data['messages'] = new_messages
                migrated = True
        
        if migrated:
            print("🔄 Migrated chat history to new message format")
            self.save_chat_history()

    def save_chat_history(self):
        """Save chat history to file"""
        try:
            CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'sessions': self.chat_sessions,
                'current_session_id': self.current_session_id,
                'last_updated': datetime.now().isoformat()
            }
            with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving chat history: {e}")

    def create_new_session(self) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        self.chat_sessions[session_id] = {
            'id': session_id,
            'title': f"Chat {len(self.chat_sessions) + 1}",
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'model': self.available_models[0] if self.available_models else 'SmallDoge/Doge-320M-Instruct'
        }
        self.current_session_id = session_id
        self.save_chat_history()
        return session_id

    def get_current_session(self) -> Dict[str, Any]:
        """Get current chat session"""
        if not self.current_session_id or self.current_session_id not in self.chat_sessions:
            self.create_new_session()
        return self.chat_sessions[self.current_session_id]

    def update_session_messages(self, messages: List[dict]):
        """Update current session messages"""
        session = self.get_current_session()
        session['messages'] = messages
        session['updated_at'] = datetime.now().isoformat()
        self.save_chat_history()
    
    def load_models(self) -> List[str]:
        """Load available models from backend and merge with local list"""
        try:
            backend_models = self.api_client.get_models()
            # Merge backend models with our local available models, keeping unique ones
            all_models = list(set(self.available_models + backend_models))
            self.available_models = all_models
            print(f"📋 Loaded {len(self.available_models)} total models: {self.available_models}")
            return self.available_models
        except Exception as e:
            print(f"⚠️ Error loading backend models: {e}")
            print(f"📋 Using local models: {self.available_models}")
            return self.available_models  # Return local list as fallback
    
    def search_huggingface_models(self, tags: str = "", task: str = "", query: str = "", limit: int = 20):
        """Search HuggingFace models with enhanced filtering"""
        try:
            # Parse tags
            tag_list = []
            if tags.strip():
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
            
            # Call backend API
            search_results = self.api_client.search_huggingface_models(
                tags=tag_list if tag_list else None,
                task=task if task else None,
                query=query if query else None,
                limit=limit
            )
            
            # Debug: Print the actual response structure
            print(f"🔍 Search API Response: {search_results}")
            
            self.search_results = search_results.get('results', [])
            error_msg = search_results.get('error', '')
            
            # Debug: Print first result structure if available
            if self.search_results:
                print(f"🔍 First result structure: {self.search_results[0]}")
            
            # Format results for display
            if not self.search_results:
                return "No models found. Try different search terms.", [], []
            
            # Create formatted display with error notice if applicable
            results_html = f"<div class='search-results'>"
            
            if error_msg:
                results_html += f"""
                <div style='background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin-bottom: 12px;'>
                    <strong>⚠️ Notice:</strong> {error_msg}
                </div>
                """
            
            results_html += f"<h4>🔍 Found {len(self.search_results)} models:</h4>"
            model_choices = []
            model_info = []
            
            for i, model in enumerate(self.search_results[:10]):  # Show top 10
                # Handle different possible field names for model ID - be more flexible
                model_id = (model.get('model_id') or 
                           model.get('id') or 
                           model.get('modelId') or 
                           model.get('name') or 
                           f'Model_{i+1}')
                
                downloads = model.get('downloads', 0)
                pipeline_tag = model.get('pipeline_tag') or model.get('task', 'unknown')
                tags_list = model.get('tags', [])
                likes = model.get('likes', 0)
                description = model.get('description', '')
                
                # Ensure tags_list is actually a list
                if not isinstance(tags_list, list):
                    tags_list = []
                
                # Format downloads number
                if downloads >= 1000000:
                    downloads_str = f"{downloads/1000000:.1f}M"
                elif downloads >= 1000:
                    downloads_str = f"{downloads/1000:.1f}K"
                else:
                    downloads_str = str(downloads)
                
                # Get first few tags for display
                display_tags = tags_list[:3] if len(tags_list) > 0 else ['no-tags']
                
                results_html += f"""
                <div class='model-result' style='margin: 8px 0; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #f9fafb;'>
                    <strong>📦 {model_id}</strong><br>
                    <small>📊 {downloads_str} downloads | ❤️ {likes} likes | 🏷️ {pipeline_tag}</small><br>
                    <small>🔖 Tags: {', '.join(display_tags)}{'...' if len(tags_list) > 3 else ''}</small>
                    {f'<br><small>📝 {description[:100]}{"..." if len(description) > 100 else ""}</small>' if description else ''}
                </div>
                """
                
                model_choices.append(model_id)
                model_info.append(f"{model_id} ({downloads_str} downloads)")
            
            results_html += "</div>"
            
            return results_html, model_choices, model_info
            
        except Exception as e:
            error_msg = f"Error searching models: {str(e)}"
            print(f"❌ Search error: {error_msg}")
            print(f"❌ Exception details: {type(e).__name__}: {e}")
            return error_msg, [], []
    
    def load_selected_model(self, model_id: str):
        """Load a selected HuggingFace model and add it to available models"""
        if not model_id:
            return "❌ Please select a model first", []
        
        try:
            # Check compatibility first
            compatibility = self.api_client.check_model_compatibility(model_id)
            
            if not compatibility.get('compatible', False):
                issues = compatibility.get('issues', [])
                warnings = compatibility.get('warnings', [])
                error_details = []
                if issues:
                    error_details.extend([f"Issues: {', '.join(issues)}"])
                if warnings:
                    error_details.extend([f"Warnings: {', '.join(warnings)}"])
                return f"❌ Model {model_id} is not compatible: {'; '.join(error_details)}", self.available_models
            
            # Attempt to load the model
            result = self.api_client.load_huggingface_model(model_id)
            
            if result.get('success', False):
                # Add the model to available models if not already present
                if model_id not in self.available_models:
                    self.available_models.append(model_id)
                    print(f"✅ Added {model_id} to available models list")
                
                # Refresh available models from backend to ensure consistency
                try:
                    backend_models = self.api_client.get_models()
                    # Merge with our local list, keeping unique models
                    all_models = list(set(self.available_models + backend_models))
                    self.available_models = all_models
                except Exception as e:
                    print(f"⚠️ Could not refresh backend models: {e}")
                    # Use our local list
                    pass
                
                return f"✅ Successfully loaded {model_id}! Model added to available models.", self.available_models
            else:
                error = result.get('error', 'Unknown error')
                error_details = []
                
                # Parse different types of errors
                if 'not found' in error.lower():
                    error_details.append("Model not found on HuggingFace Hub")
                elif 'connection' in error.lower():
                    error_details.append("Connection error - check internet connection")
                elif 'authentication' in error.lower() or 'gated' in error.lower():
                    error_details.append("Model requires authentication or is gated")
                elif 'memory' in error.lower() or 'oom' in error.lower():
                    error_details.append("Insufficient memory to load model")
                elif 'format' in error.lower():
                    error_details.append("Unsupported model format")
                else:
                    error_details.append(f"Backend error: {error}")
                
                return f"❌ Failed to load {model_id}: {'; '.join(error_details)}", self.available_models
        
        except Exception as e:
            error_msg = str(e)
            if 'connection' in error_msg.lower():
                return f"❌ Connection error: Backend may not be running. Please start the backend server.", self.available_models
            elif 'timeout' in error_msg.lower():
                return f"❌ Request timeout: Model loading is taking too long, try again later.", self.available_models
            else:
                return f"❌ Error loading model: {error_msg}", self.available_models

    def remove_selected_model(self, model_id: str):
        """Remove a selected model from available models list"""
        if not model_id:
            return "❌ Please select a model to remove", []
        
        # Prevent removal of core SmallDoge models
        core_models = ["SmallDoge/Doge-160M", "SmallDoge/Doge-60M", "SmallDoge/Doge-160M-Instruct"]
        if model_id in core_models:
            return f"❌ Cannot remove core SmallDoge model: {model_id}", self.available_models
        
        try:
            # Call backend to remove the model
            result = self.api_client.remove_model(model_id)
            
            if result.get('success', False):
                # Remove from local available models list
                if model_id in self.available_models:
                    self.available_models.remove(model_id)
                    print(f"✅ Removed {model_id} from local available models list")
                
                # Refresh available models from backend to ensure consistency
                try:
                    backend_models = self.api_client.get_models()
                    # Update our local list with backend models
                    self.available_models = backend_models
                except Exception as e:
                    print(f"⚠️ Could not refresh backend models: {e}")
                    # Keep our local list
                    pass
                
                message = result.get('message', f'Successfully removed {model_id}')
                return f"✅ {message}", self.available_models
            else:
                error = result.get('error', 'Unknown error')
                return f"❌ Failed to remove {model_id}: {error}", self.available_models
        
        except Exception as e:
            error_msg = str(e)
            if 'connection' in error_msg.lower():
                return f"❌ Connection error: Backend may not be running. Please start the backend server.", self.available_models
            elif 'timeout' in error_msg.lower():
                return f"❌ Request timeout: Model removal is taking too long, try again later.", self.available_models
            else:
                return f"❌ Error removing model: {error_msg}", self.available_models

    def get_quick_tag_suggestions(self):
        """Get popular tag suggestions for quick search"""
        return [
            ["text-generation", "conversational"],
            ["question-answering", "chat"],
            ["code-generation", "instruction-following"],
            ["summarization", "small-model"],
            ["translation", "multilingual"],
            ["fine-tuned", "efficient"]
        ]
    
    def chat_completion_streaming(
        self,
        message: str,
        history: List[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> Generator[tuple[str, List[dict]], None, None]:
        """Generate streaming chat completion"""
        if not message.strip():
            yield "", history
            return

        # Prepare messages for API
        messages = []

        # Add chat history - convert from Gradio messages format
        for msg in history:
            if msg.get("role") == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg.get("role") == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})

        # Add current message
        messages.append({"role": "user", "content": message})

        # Add user message to history immediately
        new_history = history + [{"role": "user", "content": message}]
        yield "", new_history

        try:
            # Use enhanced API client for streaming
            assistant_message = ""

            for token in self.api_client.chat_completion_streaming(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            ):
                assistant_message += token

                # Update history with streaming content
                updated_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": assistant_message}
                ]
                yield "", updated_history

            # Save final history
            final_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": assistant_message}
            ]
            self.update_session_messages(final_history)
            yield "", final_history

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            error_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg}
            ]
            yield "", error_history
    
    def clear_chat(self) -> List[dict]:
        """Clear current chat session"""
        if self.current_session_id:
            session = self.get_current_session()
            session['messages'] = []
            self.save_chat_history()
        return []

    def new_chat(self) -> tuple[List[dict], str]:
        """Create a new chat session"""
        session_id = self.create_new_session()
        return [], f"New chat created: {session_id[:8]}"

    def get_chat_sessions_list(self) -> List[str]:
        """Get list of chat sessions for dropdown"""
        sessions = []
        for session_id, session_data in self.chat_sessions.items():
            title = session_data.get('title', f"Chat {session_id[:8]}")
            sessions.append(f"{title} ({session_id[:8]})")
        return sessions if sessions else ["No chats available"]

    def load_chat_session(self, session_selection) -> List[dict]:
        """Load a specific chat session"""
        # Handle case where session_selection might be a list
        if isinstance(session_selection, list):
            if not session_selection or session_selection[0] == "No chats available":
                return []
            session_selection = session_selection[0]
        
        if not session_selection or session_selection == "No chats available":
            return []

        # Extract session ID from selection
        try:
            session_id = session_selection.split('(')[-1].rstrip(')')
        except (AttributeError, IndexError):
            return []

        # Find full session ID
        for full_id in self.chat_sessions:
            if full_id.startswith(session_id):
                self.current_session_id = full_id
                session = self.chat_sessions[full_id]
                return session.get('messages', [])

        return []

    def export_chat(self) -> str:
        """Export current chat session"""
        session = self.get_current_session()
        export_data = {
            'session_id': session['id'],
            'title': session['title'],
            'messages': session['messages'],
            'created_at': session.get('created_at'),
            'exported_at': datetime.now().isoformat()
        }

        filename = f"chat_export_{session['id'][:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return f"Chat exported to {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"

    def check_backend_health(self) -> tuple[str, str]:
        """Check backend health and return status"""
        try:
            if self.api_client.health_check():
                return "🟢 Backend: Connected", "status-online"
            else:
                return "🔴 Backend: Disconnected", "status-offline"
        except Exception as e:
            return f"🔴 Backend: Error - {str(e)}", "status-offline"

    def get_model_info(self, model_id: str) -> str:
        """Get model information for display"""
        try:
            info = self.api_client.get_model_info(model_id)
            status = self.api_client.get_model_status(model_id)

            # Handle cases where endpoints might not be available
            if info and info.get('status') != 'error' and status and status.get('status') != 'error':
                status_text = status.get('status', 'Unknown')
                context_length = info.get('context_length', status.get('context_length', 'Unknown'))
                capabilities = info.get('capabilities', ['text-generation', 'chat-completion'])
                
                if isinstance(capabilities, list):
                    capabilities_text = ', '.join(capabilities)
                else:
                    capabilities_text = str(capabilities)
                
                return f"""
                <div class="model-info">
                    <strong>Model:</strong> {model_id}<br>
                    <strong>Status:</strong> {status_text}<br>
                    <strong>Context Length:</strong> {context_length}<br>
                    <strong>Capabilities:</strong> {capabilities_text}
                </div>
                """
            else:
                # Fallback display when endpoints are not available
                return f"""
                <div class="model-info">
                    <strong>Model:</strong> {model_id}<br>
                    <strong>Status:</strong> Loaded<br>
                    <strong>Context Length:</strong> 2048<br>
                    <strong>Capabilities:</strong> text-generation, chat-completion
                </div>
                """
        except Exception as e:
            return f"""
            <div class="model-info">
                <strong>Model:</strong> {model_id}<br>
                <strong>Status:</strong> Loaded<br>
                <em>Model info unavailable</em>
            </div>
            """
    
    def load_initial_hf_data(self):
        """Load initial HuggingFace data"""
        try:
            # Load task categories
            categories_data = self.api_client.get_task_categories()
            self.task_categories = categories_data.get('categories', [])
            print(f"✅ Loaded {len(self.task_categories)} task categories")
            
            # Load model families
            families_data = self.api_client.get_model_families()
            self.model_families = families_data.get('families', {})
            print(f"✅ Loaded {len(self.model_families)} model families")
            
            # Load some featured models for quick access
            featured_data = self.api_client.get_featured_models()
            self.featured_models = featured_data.get('featured', {})
            print(f"✅ Loaded featured models from {len(self.featured_models)} categories")
            
        except Exception as e:
            print(f"⚠️ Could not load HuggingFace data: {e}")
            self.task_categories = []
            self.model_families = {}
            self.featured_models = {}

    def create_interface(self) -> gr.Blocks:
        """Create the enhanced Gradio interface"""
        # Custom CSS for better UI/UX matching open-webui patterns
        custom_css = """
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto;
        }
        .chat-container {
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            background: #ffffff;
        }
        .sidebar {
            background: #f8fafc;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #e5e7eb;
        }
        .message-input {
            border-radius: 8px;
            border: 1px solid #d1d5db;
        }
        .send-button {
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            font-weight: 600;
        }
        .model-info {
            background: #f0f9ff;
            border: 1px solid #0ea5e9;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }
        .search-results {
            max-height: 400px;
            overflow-y: auto;
            margin: 8px 0;
        }
        .model-result {
            margin: 8px 0;
            padding: 12px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #f9fafb;
            transition: all 0.2s;
        }
        .model-result:hover {
            background: #f3f4f6;
            border-color: #d1d5db;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background-color: #10b981; }
        .status-offline { background-color: #ef4444; }
        .tag-btn {
            margin: 2px;
            font-size: 12px;
            padding: 4px 8px;
        }
        .search-section {
            background: #fafbfc;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        }
        .quick-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin: 8px 0;
        }
        """

        with gr.Blocks(
            title="🐕 SmallDoge WebUI - Open Source AI Chat",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:

            # Header with status
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown(
                        """
                        # 🐕 SmallDoge WebUI
                        **Open Source AI Chat Platform** - Real-time streaming responses, no authentication required!
                        """
                    )
                with gr.Column(scale=1):
                    status_display = gr.HTML(
                        '<div class="model-info">🟢 <strong>Status:</strong> Ready</div>'
                    )

            # Main interface layout
            with gr.Row():
                # Sidebar for chat management and settings
                with gr.Column(scale=1, elem_classes=["sidebar"]):
                    gr.Markdown("### 💬 Chat Management")

                    new_chat_btn = gr.Button("🆕 New Chat", variant="secondary", size="sm")

                    chat_sessions_dropdown = gr.Dropdown(
                        label="📋 Chat History",
                        choices=self.get_chat_sessions_list(),
                        value=None,
                        interactive=True,
                        allow_custom_value=True
                    )

                    with gr.Row():
                        export_btn = gr.Button("📤 Export", size="sm", scale=1)
                        clear_btn = gr.Button("🗑️ Clear", size="sm", scale=1)

                    gr.Markdown("### ⚙️ Model Settings")

                    model_dropdown = gr.Dropdown(
                        label="🤖 Model",
                        choices=self.available_models,
                        value=self.available_models[0] if self.available_models else "SmallDoge/Doge-320M-Instruct",
                        interactive=True
                    )

                    with gr.Accordion("🎛️ Advanced Parameters", open=False):
                        temperature_slider = gr.Slider(
                            label="🌡️ Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            info="Controls randomness in responses"
                        )

                        max_tokens_slider = gr.Slider(
                            label="📏 Max Tokens",
                            minimum=1,
                            maximum=4096,
                            value=2048,
                            step=1,
                            info="Maximum response length"
                        )

                        top_p_slider = gr.Slider(
                            label="🎯 Top P",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            info="Controls diversity via nucleus sampling"
                        )

                    refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")

                    # Model management buttons
                    with gr.Row():
                        remove_model_btn = gr.Button("🗑️ Remove Model", size="sm", variant="stop")
                    
                    remove_result_display = gr.HTML("")

                    # Model information display
                    model_info_display = gr.HTML("")

                    # Enhanced HuggingFace Model Search Section
                    gr.Markdown("### 🔍 Model Discovery")
                    
                    with gr.Accordion("🚀 Search HuggingFace Models", open=True):
                        gr.Markdown("""
                        **🏷️ Tag-Based Search**: Find models by specific capabilities
                        
                        **Popular Tags**: `text-generation`, `conversational`, `chat`, `question-answering`, `code-generation`, `small-model`, `fine-tuned`
                        """)
                        
                        # Quick tag buttons
                        gr.Markdown("**⚡ Quick Tags:**")
                        with gr.Row():
                            tag_btn_1 = gr.Button("text-generation", size="sm", variant="secondary")
                            tag_btn_2 = gr.Button("conversational", size="sm", variant="secondary")
                        with gr.Row():
                            tag_btn_3 = gr.Button("question-answering", size="sm", variant="secondary")
                            tag_btn_4 = gr.Button("code-generation", size="sm", variant="secondary")
                        
                        # Search inputs
                        search_tags_input = gr.Textbox(
                            label="🏷️ Tags (comma-separated)",
                            placeholder="e.g., text-generation, chat, small-model",
                            value="",
                            lines=1
                        )
                        
                        search_task_dropdown = gr.Dropdown(
                            label="📋 Task Type",
                            choices=[
                                "", "text-generation", "conversational", "question-answering", 
                                "summarization", "translation", "text-classification",
                                "token-classification", "fill-mask", "feature-extraction"
                            ],
                            value="",
                            interactive=True
                        )
                        
                        search_query_input = gr.Textbox(
                            label="🔎 Keyword Search",
                            placeholder="e.g., SmallDoge, chat, instruct",
                            value="",
                            lines=1
                        )
                        
                        search_limit_slider = gr.Slider(
                            label="📊 Max Results",
                            minimum=5,
                            maximum=50,
                            value=20,
                            step=5
                        )
                        
                        search_btn = gr.Button("🔍 Search Models", variant="primary", size="sm")
                        
                        # Search results display
                        search_results_display = gr.HTML("")
                        
                        # Model selection from search results
                        search_results_dropdown = gr.Dropdown(
                            label="📦 Select Model to Load",
                            choices=[],
                            value=None,
                            interactive=True,
                            visible=False
                        )
                        
                        load_model_btn = gr.Button("📥 Load Selected Model", variant="secondary", size="sm", visible=False)
                        load_result_display = gr.HTML("")

                # Main chat area
                with gr.Column(scale=3, elem_classes=["chat-container"]):
                    chatbot = gr.Chatbot(
                        label="💬 Chat",
                        height=600,
                        show_label=True,
                        container=True,
                        show_copy_button=True,
                        layout="panel",
                        type="messages"
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="",
                            placeholder="💭 Type your message here... (Press Enter to send)",
                            scale=5,
                            lines=2,
                            max_lines=6,
                            elem_classes=["message-input"]
                        )
                        send_btn = gr.Button(
                            "🚀 Send",
                            variant="primary",
                            scale=1,
                            elem_classes=["send-button"]
                        )

                    # Status and info row
                    with gr.Row():
                        typing_indicator = gr.HTML("")
                        message_info = gr.HTML("")

            # Event handlers with enhanced functionality
            def handle_refresh_models():
                """Refresh available models"""
                try:
                    models = self.load_models()
                    return gr.update(choices=models, value=models[0] if models else "SmallDoge/Doge-160M")
                except Exception as e:
                    print(f"Error refreshing models: {e}")
                    return gr.update(choices=["SmallDoge/Doge-160M"], value="SmallDoge/Doge-160M")

            def handle_new_chat():
                """Create new chat session"""
                try:
                    history, message = self.new_chat()
                    sessions = self.get_chat_sessions_list()
                    return history, gr.update(choices=sessions), f"✨ {message}"
                except Exception as e:
                    print(f"Error creating new chat: {e}")
                    return [], gr.update(), "❌ Failed to create new chat"

            def handle_load_chat(session_selection):
                """Load selected chat session"""
                try:
                    history = self.load_chat_session(session_selection)
                    return history, "📂 Chat loaded successfully"
                except Exception as e:
                    print(f"Error loading chat: {e}")
                    return [], "❌ Failed to load chat"

            def handle_export_chat():
                """Export current chat"""
                try:
                    result = self.export_chat()
                    return f"📤 {result}"
                except Exception as e:
                    print(f"Error exporting chat: {e}")
                    return "❌ Export failed"

            def handle_clear_chat():
                """Clear current chat"""
                try:
                    history = self.clear_chat()
                    return history, "🗑️ Chat cleared"
                except Exception as e:
                    print(f"Error clearing chat: {e}")
                    return [], "❌ Failed to clear chat"

            def show_typing_indicator():
                """Show typing indicator"""
                return "🤖 SmallDoge is thinking..."

            def hide_typing_indicator():
                """Hide typing indicator"""
                return ""

            def update_status():
                """Update backend status"""
                try:
                    status_text, status_class = self.check_backend_health()
                    return f'<div class="model-info"><span class="status-indicator {status_class}"></span>{status_text}</div>'
                except Exception as e:
                    print(f"Error updating status: {e}")
                    return '<div class="model-info">🔴 <strong>Status:</strong> Error</div>'

            def update_model_info(model_id):
                """Update model information display"""
                try:
                    if model_id:
                        return self.get_model_info(model_id)
                    return ""
                except Exception as e:
                    print(f"Error updating model info: {e}")
                    return f'<div class="model-info"><strong>Model:</strong> {model_id or "Unknown"}</div>'

            # Enhanced Model Search Event Handlers
            def handle_search_models(tags, task, query, limit):
                """Handle HuggingFace model search"""
                try:
                    results_html, model_choices, model_info = self.search_huggingface_models(
                        tags=tags, task=task, query=query, limit=int(limit)
                    )
                    
                    if model_choices:
                        return (
                            results_html,
                            gr.update(choices=model_choices, value=None, visible=True),
                            gr.update(visible=True),
                            ""
                        )
                    else:
                        return (
                            results_html,
                            gr.update(choices=[], visible=False),
                            gr.update(visible=False),
                            ""
                        )
                except Exception as e:
                    error_msg = f"❌ Search error: {str(e)}"
                    return (
                        error_msg,
                        gr.update(choices=[], visible=False),
                        gr.update(visible=False),
                        ""
                    )

            def handle_quick_tag(tag_name, current_tags):
                """Handle quick tag button clicks"""
                if not current_tags:
                    return tag_name
                elif tag_name not in current_tags:
                    return f"{current_tags}, {tag_name}"
                else:
                    return current_tags

            def handle_load_model(model_id):
                """Handle loading a selected model"""
                try:
                    result_msg, updated_models = self.load_selected_model(model_id)
                    
                    if "✅" in result_msg:  # Success
                        return (
                            result_msg,
                            gr.update(choices=updated_models, value=model_id)
                        )
                    else:  # Error
                        return (
                            result_msg,
                            gr.update()  # Don't change model dropdown on error
                        )
                except Exception as e:
                    return (
                        f"❌ Error loading model: {str(e)}",
                        gr.update()
                    )

            def handle_remove_model(model_id):
                """Handle removing a selected model"""
                try:
                    result_msg, updated_models = self.remove_selected_model(model_id)
                    
                    if "✅" in result_msg:  # Success
                        # Set dropdown to first available model if current model was removed
                        new_value = updated_models[0] if updated_models else "SmallDoge/Doge-160M"
                        return (
                            result_msg,
                            gr.update(choices=updated_models, value=new_value)
                        )
                    else:  # Error
                        return (
                            result_msg,
                            gr.update()  # Don't change model dropdown on error
                        )
                except Exception as e:
                    return (
                        f"❌ Error removing model: {str(e)}",
                        gr.update()
                    )

            # Connect events with improved UX and debugging
            print("🔗 Connecting event handlers...")
            
            try:
                refresh_models_btn.click(
                    handle_refresh_models,
                    outputs=[model_dropdown]
                )
                print("✅ Refresh models button connected")
            except Exception as e:
                print(f"❌ Error connecting refresh models button: {e}")

            try:
                # Remove model functionality
                remove_model_btn.click(
                    handle_remove_model,
                    inputs=[model_dropdown],
                    outputs=[remove_result_display, model_dropdown]
                )
                print("✅ Remove model button connected")
            except Exception as e:
                print(f"❌ Error connecting remove model button: {e}")

            try:
                # Update model info when model changes
                model_dropdown.change(
                    update_model_info,
                    inputs=[model_dropdown],
                    outputs=[model_info_display]
                )
                print("✅ Model dropdown change connected")
            except Exception as e:
                print(f"❌ Error connecting model dropdown: {e}")

            try:
                new_chat_btn.click(
                    handle_new_chat,
                    outputs=[chatbot, chat_sessions_dropdown, message_info]
                )
                print("✅ New chat button connected")
            except Exception as e:
                print(f"❌ Error connecting new chat button: {e}")

            try:
                chat_sessions_dropdown.change(
                    handle_load_chat,
                    inputs=[chat_sessions_dropdown],
                    outputs=[chatbot, message_info]
                )
                print("✅ Chat sessions dropdown connected")
            except Exception as e:
                print(f"❌ Error connecting chat sessions dropdown: {e}")

            try:
                export_btn.click(
                    handle_export_chat,
                    outputs=[message_info]
                )
                print("✅ Export button connected")
            except Exception as e:
                print(f"❌ Error connecting export button: {e}")

            try:
                clear_btn.click(
                    handle_clear_chat,
                    outputs=[chatbot, message_info]
                )
                print("✅ Clear button connected")
            except Exception as e:
                print(f"❌ Error connecting clear button: {e}")

            try:
                # Streaming chat completion with typing indicators
                send_btn.click(
                    show_typing_indicator,
                    outputs=[typing_indicator]
                ).then(
                    self.chat_completion_streaming,
                    inputs=[
                        msg_input,
                        chatbot,
                        model_dropdown,
                        temperature_slider,
                        max_tokens_slider,
                        top_p_slider
                    ],
                    outputs=[msg_input, chatbot]
                ).then(
                    hide_typing_indicator,
                    outputs=[typing_indicator]
                )
                print("✅ Send button connected")
            except Exception as e:
                print(f"❌ Error connecting send button: {e}")

            try:
                msg_input.submit(
                    show_typing_indicator,
                    outputs=[typing_indicator]
                ).then(
                    self.chat_completion_streaming,
                    inputs=[
                        msg_input,
                        chatbot,
                        model_dropdown,
                        temperature_slider,
                        max_tokens_slider,
                        top_p_slider
                    ],
                    outputs=[msg_input, chatbot]
                ).then(
                    hide_typing_indicator,
                    outputs=[typing_indicator]
                )
                print("✅ Message input submit connected")
            except Exception as e:
                print(f"❌ Error connecting message input submit: {e}")

            # Model search functionality
            try:
                search_btn.click(
                    handle_search_models,
                    inputs=[search_tags_input, search_task_dropdown, search_query_input, search_limit_slider],
                    outputs=[search_results_display, search_results_dropdown, load_model_btn, load_result_display]
                )
                print("✅ Search button connected")
            except Exception as e:
                print(f"❌ Error connecting search button: {e}")

            # Quick tag buttons
            try:
                tag_btn_1.click(
                    lambda current: handle_quick_tag("text-generation", current),
                    inputs=[search_tags_input],
                    outputs=[search_tags_input]
                )
                tag_btn_2.click(
                    lambda current: handle_quick_tag("conversational", current),
                    inputs=[search_tags_input],
                    outputs=[search_tags_input]
                )
                tag_btn_3.click(
                    lambda current: handle_quick_tag("question-answering", current),
                    inputs=[search_tags_input],
                    outputs=[search_tags_input]
                )
                tag_btn_4.click(
                    lambda current: handle_quick_tag("code-generation", current),
                    inputs=[search_tags_input],
                    outputs=[search_tags_input]
                )
                print("✅ Quick tag buttons connected")
            except Exception as e:
                print(f"❌ Error connecting quick tag buttons: {e}")

            try:
                # Model loading functionality
                load_model_btn.click(
                    handle_load_model,
                    inputs=[search_results_dropdown],
                    outputs=[load_result_display, model_dropdown]
                )
                print("✅ Load model button connected")
            except Exception as e:
                print(f"❌ Error connecting load model button: {e}")

            try:
                # Periodic status updates (simplified) - no timer for now
                def periodic_status_update():
                    return update_status()

                print("✅ Status update function ready (manual updates only)")
            except Exception as e:
                print(f"❌ Error setting up status update: {e}")
                
            print("🎯 All event handlers setup complete")

            # Load initial data
            def load_initial_data():
                """Load initial data for the interface"""
                try:
                    # Refresh models
                    models = self.load_models()
                    default_model = models[0] if models else self.available_models[0] if self.available_models else "SmallDoge/Doge-320M-Instruct"
                    model_update = gr.update(choices=models, value=default_model)
                    
                    # Get chat sessions
                    sessions = self.get_chat_sessions_list()
                    sessions_update = gr.update(choices=sessions)
                    
                    # Update status
                    status_text, status_class = self.check_backend_health()
                    status_html = f'<div class="model-info"><span class="status-indicator {status_class}"></span>{status_text}</div>'
                    
                    # Update model info
                    model_info_html = self.get_model_info(default_model)
                    
                    return model_update, sessions_update, status_html, model_info_html
                except Exception as e:
                    print(f"Error loading initial data: {e}")
                    fallback_model = self.available_models[0] if self.available_models else "SmallDoge/Doge-320M-Instruct"
                    return (
                        gr.update(choices=self.available_models, value=fallback_model),
                        gr.update(choices=["No chats available"]),
                        '<div class="model-info">🔴 <strong>Status:</strong> Error</div>',
                        f'<div class="model-info"><strong>Model:</strong> {fallback_model}</div>'
                    )

            interface.load(
                load_initial_data,
                outputs=[model_dropdown, chat_sessions_dropdown, status_display, model_info_display]
            )

            # Skip automatic status timer for now to avoid Gradio issues
            print("⏰ Automatic status updates disabled to prevent Gradio validation errors")
        
        return interface


def main():
    """Main entry point"""
    print("🐕 Starting SmallDoge WebUI Frontend...")
    print("=" * 50)

    # Create the WebUI instance
    webui = SmallDogeWebUI()

    # Create and launch the interface
    interface = webui.create_interface()

    print("🚀 Launching Gradio interface...")
    print(f"📡 Backend URL: {BACKEND_URL}")
    print(f"💾 Chat history: {CHAT_HISTORY_FILE}")
    print("=" * 50)

    # Launch the interface with simplified configuration
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
