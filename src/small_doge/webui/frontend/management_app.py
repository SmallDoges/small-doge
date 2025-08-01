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
Dataset and Training Management UI for SmallDoge WebUI
Provides a simple interface for dataset and training management
"""

import gradio as gr
import requests
import os
from typing import Dict, Any, List

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def create_datasets_interface():
    """Create the datasets management interface"""
    
    def load_available_datasets():
        """Load available datasets from API"""
        try:
            response = requests.get(f"{BACKEND_URL}/api/v1/datasets/available")
            if response.status_code == 200:
                data = response.json()
                datasets = data.get("datasets", {})
                
                pretrain_datasets = datasets.get("pretrain", [])
                finetune_datasets = datasets.get("finetune", [])
                
                html = """
                <div style='background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    <h3>📊 Available Datasets</h3>
                    <div style='display: flex; gap: 20px;'>
                        <div style='flex: 1;'>
                            <h4>🔤 Pretrain Datasets:</h4>
                            <ul>
                """
                for dataset in pretrain_datasets:
                    html += f"<li>📊 {dataset}</li>"
                
                html += """
                            </ul>
                        </div>
                        <div style='flex: 1;'>
                            <h4>💬 Finetune Datasets:</h4>
                            <ul>
                """
                for dataset in finetune_datasets:
                    html += f"<li>🎯 {dataset}</li>"
                
                html += """
                            </ul>
                        </div>
                    </div>
                </div>
                """
                
                return html, gr.update(choices=pretrain_datasets + finetune_datasets)
            else:
                return f"<div style='color: red; padding: 10px;'>❌ Error loading datasets: {response.status_code}</div>", gr.update()
        except Exception as e:
            return f"<div style='color: red; padding: 10px;'>❌ Error: {str(e)}</div>", gr.update()
    
    def load_downloaded_datasets():
        """Load downloaded datasets from API"""
        try:
            response = requests.get(f"{BACKEND_URL}/api/v1/datasets/downloaded")
            if response.status_code == 200:
                data = response.json()
                datasets = data.get("downloaded_datasets", [])
                
                if not datasets:
                    return "<div style='background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin: 8px 0;'>📂 No datasets downloaded yet</div>"
                
                html = """
                <div style='background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    <h3>📋 Downloaded Datasets</h3>
                    <ul>
                """
                for dataset in datasets:
                    html += f"<li>✅ {dataset}</li>"
                html += """
                    </ul>
                </div>
                """
                
                return html
            else:
                return f"<div style='color: red; padding: 10px;'>❌ Error: {response.status_code}</div>"
        except Exception as e:
            return f"<div style='color: red; padding: 10px;'>❌ Error: {str(e)}</div>"
    
    def download_dataset_handler(dataset_name, save_dir, cache_dir):
        """Handle dataset download"""
        if not dataset_name:
            return "❌ Please select a dataset to download"
        
        try:
            payload = {
                "dataset_name": dataset_name,
                "save_dir": save_dir,
                "cache_dir": cache_dir,
                "num_proc": 1
            }
            
            response = requests.post(f"{BACKEND_URL}/api/v1/datasets/download", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return f"<div style='color: green; padding: 10px; background: #f0fdf4; border-radius: 8px;'>✅ {result.get('message', 'Download completed')}</div>"
            else:
                error_data = response.json()
                return f"<div style='color: red; padding: 10px; background: #fef2f2; border-radius: 8px;'>❌ Download failed: {error_data.get('detail', 'Unknown error')}</div>"
        except Exception as e:
            return f"<div style='color: red; padding: 10px; background: #fef2f2; border-radius: 8px;'>❌ Error: {str(e)}</div>"
    
    # UI Components
    with gr.Column():
        gr.Markdown("## 📊 Dataset Management")
        
        # Available datasets display
        available_datasets_display = gr.HTML("")
        refresh_datasets_btn = gr.Button("🔄 Refresh Available Datasets", variant="secondary", size="sm")
        
        # Download section
        with gr.Group():
            gr.Markdown("### Download Dataset")
            
            dataset_selection = gr.Dropdown(
                label="Select Dataset",
                choices=[],
                interactive=True
            )
            
            with gr.Row():
                save_dir_input = gr.Textbox(
                    label="Save Directory",
                    value="./data/datasets",
                    placeholder="Directory to save datasets"
                )
                cache_dir_input = gr.Textbox(
                    label="Cache Directory", 
                    value="./data/cache",
                    placeholder="Directory for caching"
                )
            
            download_btn = gr.Button("📥 Download Dataset", variant="primary")
            download_status = gr.HTML("")
        
        # Downloaded datasets display
        downloaded_datasets_display = gr.HTML("")
        refresh_downloaded_btn = gr.Button("🔄 Refresh Downloaded", variant="secondary", size="sm")
    
    # Event handlers
    refresh_datasets_btn.click(
        fn=load_available_datasets,
        outputs=[available_datasets_display, dataset_selection]
    )
    
    refresh_downloaded_btn.click(
        fn=load_downloaded_datasets,
        outputs=[downloaded_datasets_display]
    )
    
    download_btn.click(
        fn=download_dataset_handler,
        inputs=[dataset_selection, save_dir_input, cache_dir_input],
        outputs=[download_status]
    )
    
    # Load initial data
    load_available_datasets()


def create_training_interface():
    """Create the training management interface"""
    
    def load_training_jobs():
        """Load training jobs"""
        try:
            response = requests.get(f"{BACKEND_URL}/api/v1/training/jobs")
            if response.status_code == 200:
                data = response.json()
                jobs = data.get("jobs", [])
                
                if not jobs:
                    return "<div style='background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin: 8px 0;'>🎯 No training jobs found</div>"
                
                html = """
                <div style='background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px; padding: 12px; margin: 8px 0;'>
                    <h3>📊 Training Jobs</h3>
                    <ul>
                """
                for job in jobs:
                    status_icon = "🟢" if job["status"] == "running" else "⏸️" if job["status"] == "stopped" else "🔴"
                    html += f"<li>{status_icon} <strong>{job['name']}</strong> - {job['status']}</li>"
                html += """
                    </ul>
                </div>
                """
                
                return html
            else:
                return f"<div style='color: red; padding: 10px;'>❌ Error: {response.status_code}</div>"
        except Exception as e:
            return f"<div style='color: red; padding: 10px;'>❌ Error: {str(e)}</div>"
    
    def start_training_handler(training_type, model_type, dataset_path, output_dir):
        """Handle training start"""
        try:
            payload = {
                "training_type": training_type,
                "model_type": model_type,
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "model_config": f"./recipes/{model_type}/Doge-160M/config_full.yaml"
            }
            
            response = requests.post(f"{BACKEND_URL}/api/v1/training/start", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return f"<div style='color: green; padding: 10px; background: #f0fdf4; border-radius: 8px;'>✅ {result.get('message', 'Training started')}</div>"
            else:
                error_data = response.json()
                return f"<div style='color: red; padding: 10px; background: #fef2f2; border-radius: 8px;'>❌ Training failed to start: {error_data.get('detail', 'Unknown error')}</div>"
        except Exception as e:
            return f"<div style='color: red; padding: 10px; background: #fef2f2; border-radius: 8px;'>❌ Error: {str(e)}</div>"
    
    # UI Components
    with gr.Column():
        gr.Markdown("## 🎯 Training Management")
        
        # Training configuration
        with gr.Group():
            gr.Markdown("### Training Configuration")
            
            with gr.Row():
                training_type_dropdown = gr.Dropdown(
                    label="Training Type",
                    choices=["pretrain", "sft", "dpo", "grpo"],
                    value="pretrain",
                    interactive=True
                )
                
                model_type_dropdown = gr.Dropdown(
                    label="Model Architecture",
                    choices=["doge", "doge2"],
                    value="doge",
                    interactive=True
                )
            
            dataset_path_input = gr.Textbox(
                label="Dataset Path",
                placeholder="Path to dataset directory (e.g., ./data/datasets/fineweb-edu)",
                value="./data/datasets"
            )
            
            output_dir_input = gr.Textbox(
                label="Output Directory",
                placeholder="Directory to save trained model",
                value="./data/training"
            )
            
            start_training_btn = gr.Button("🚀 Start Training", variant="primary")
            training_status = gr.HTML("")
        
        # Training jobs display
        training_jobs_display = gr.HTML("")
        refresh_jobs_btn = gr.Button("🔄 Refresh Jobs", variant="secondary", size="sm")
    
    # Event handlers
    start_training_btn.click(
        fn=start_training_handler,
        inputs=[training_type_dropdown, model_type_dropdown, dataset_path_input, output_dir_input],
        outputs=[training_status]
    )
    
    refresh_jobs_btn.click(
        fn=load_training_jobs,
        outputs=[training_jobs_display]
    )
    
    # Load initial data
    load_training_jobs()


def create_management_interface():
    """Create the complete dataset and training management interface"""
    
    with gr.Blocks(
        title="🐕 SmallDoge WebUI - Dataset & Training Management",
        theme=gr.themes.Soft(),
    ) as interface:
        
        # Header
        gr.Markdown(
            """
            # 🐕 SmallDoge WebUI - Management Console
            **Dataset & Training Management** - Download datasets and manage training jobs with ease!
            
            > 💡 **Tip**: Start the backend server first: `python -m src.small_doge.webui.backend.start`
            """
        )
        
        # Main interface with tabs
        with gr.Tabs():
            # Datasets Tab
            with gr.TabItem("📊 Datasets", id="datasets"):
                create_datasets_interface()
            
            # Training Tab  
            with gr.TabItem("🎯 Training", id="training"):
                create_training_interface()
            
            # Information Tab
            with gr.TabItem("📖 Information", id="info"):
                gr.Markdown("""
                ## 📖 Information
                
                ### 📊 Available Datasets
                
                **Pretrain Datasets:**
                - `fineweb-edu` - Educational web content from FineWeb
                - `cosmopedia-v2` - Synthetic educational content
                - `finemath` - Mathematical content and reasoning
                
                **Finetune Datasets:**
                - `smoltalk` - Conversational data for chat training
                - `ultrafeedback_binarized` - Preference data for DPO
                - `open_thoughts` - Reasoning and thinking data
                
                ### 🎯 Training Types
                
                - **Pretrain**: Train language models from scratch or continue pretraining
                - **SFT**: Supervised fine-tuning for instruction following
                - **DPO**: Direct Preference Optimization for alignment
                - **GRPO**: Group Relative Policy Optimization for advanced alignment
                
                ### 🏗️ Model Architectures
                
                - **doge**: Original SmallDoge architecture
                - **doge2**: Enhanced SmallDoge architecture with improved performance
                
                ### 🔧 Usage Instructions
                
                1. **Download Datasets**: Select and download datasets you need for training
                2. **Configure Training**: Choose training type, model architecture, and paths
                3. **Start Training**: Launch training jobs and monitor progress
                4. **Monitor Jobs**: Check status and logs of running training jobs
                
                ### 📡 API Endpoints
                
                The management interface connects to these backend endpoints:
                - `/api/v1/datasets/*` - Dataset management
                - `/api/v1/training/*` - Training management
                """)
    
    return interface


def main():
    """Main entry point for the management interface"""
    print("🐕 Starting SmallDoge Management Interface...")
    print("=" * 50)
    
    interface = create_management_interface()
    
    print("🚀 Launching management interface...")
    print(f"📡 Backend URL: {BACKEND_URL}")
    print("🔗 Interface will be available at: http://localhost:7862")
    print("=" * 50)
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()