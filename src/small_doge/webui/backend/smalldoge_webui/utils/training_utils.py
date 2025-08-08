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
Training utilities for SmallDoge WebUI
Provides functions to wrap and execute training scripts
"""

import os
import sys
import subprocess
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


def get_available_training_types() -> List[str]:
    """Get list of available training types"""
    return ["pretrain", "sft", "dpo", "grpo"]


def get_available_models() -> List[str]:
    """Get list of available model architectures"""
    return ["doge", "doge2"]


def get_training_configs(model_type: str = "doge") -> Dict[str, List[str]]:
    """Get available training configurations for a model type"""
    configs = {
        "doge": [
            "Doge-20M",
            "Doge-40M", 
            "Doge-160M",
            "Doge-320M"
        ],
        "doge2": [
            "Doge2-20M",
            "Doge2-40M",
            "Doge2-160M", 
            "Doge2-320M"
        ]
    }
    return configs.get(model_type, [])


def check_training_requirements(
    training_type: str,
    model_type: str,
    dataset_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """Check if training requirements are met"""
    issues = []
    
    # Check dataset path
    if not os.path.exists(dataset_path):
        issues.append(f"Dataset path does not exist: {dataset_path}")
    
    # Check output directory can be created
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        issues.append(f"Cannot create output directory {output_dir}: {str(e)}")
    
    # Check model type
    if model_type not in get_available_models():
        issues.append(f"Unknown model type: {model_type}")
    
    # Check training type
    if training_type not in get_available_training_types():
        issues.append(f"Unknown training type: {training_type}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def build_training_command(
    training_type: str,
    model_type: str,
    dataset_path: str,
    output_dir: str,
    model_config: str,
    additional_args: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Build training command for execution"""
    
    # Get the path to the training script
    script_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..",  # Navigate to src/small_doge/
        "trainer",
        model_type,
        f"{training_type}.py"
    )
    script_path = os.path.abspath(script_path)
    
    cmd = [
        sys.executable,
        script_path,
        f"--dataset_name={dataset_path}",
        f"--output_dir={output_dir}",
        f"--config={model_config}"
    ]
    
    # Add additional arguments
    if additional_args:
        for key, value in additional_args.items():
            if value is not None:
                cmd.append(f"--{key}={value}")
    
    return cmd


def start_training(
    training_type: str,
    model_type: str,
    dataset_path: str,
    output_dir: str,
    model_config: str,
    additional_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Start a training job"""
    
    # Check requirements
    check_result = check_training_requirements(training_type, model_type, dataset_path, output_dir)
    if not check_result["valid"]:
        return {
            "status": "error",
            "message": "Training requirements not met",
            "issues": check_result["issues"]
        }
    
    try:
        # Build command
        cmd = build_training_command(
            training_type, model_type, dataset_path, output_dir, model_config, additional_args
        )
        
        # For WebUI, we'll start the process asynchronously and return immediately
        # In a production system, you'd want to use a proper job queue
        logger.info(f"Starting training with command: {' '.join(cmd)}")
        
        # Create a simple status file to track training
        status_file = os.path.join(output_dir, "training_status.txt")
        with open(status_file, "w") as f:
            f.write("STARTED\n")
            f.write(f"Command: {' '.join(cmd)}\n")
        
        return {
            "status": "started",
            "message": f"Training {training_type} started for {model_type}",
            "output_dir": output_dir,
            "command": " ".join(cmd),
            "status_file": status_file
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return {
            "status": "error", 
            "message": f"Failed to start training: {str(e)}"
        }


def get_training_status(output_dir: str) -> Dict[str, Any]:
    """Get status of a training job"""
    status_file = os.path.join(output_dir, "training_status.txt")
    
    if not os.path.exists(status_file):
        return {
            "status": "unknown",
            "message": "No training status found"
        }
    
    try:
        with open(status_file, "r") as f:
            lines = f.readlines()
            
        if lines:
            status = lines[0].strip()
            return {
                "status": status.lower(),
                "message": f"Training status: {status}",
                "details": "".join(lines[1:]) if len(lines) > 1 else ""
            }
        else:
            return {
                "status": "unknown",
                "message": "Empty status file"
            }
            
    except Exception as e:
        logger.error(f"Failed to read training status: {e}")
        return {
            "status": "error",
            "message": f"Failed to read status: {str(e)}"
        }


def list_training_jobs(base_output_dir: str) -> List[Dict[str, Any]]:
    """List all training jobs"""
    if not os.path.exists(base_output_dir):
        return []
    
    jobs = []
    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)
        if os.path.isdir(item_path):
            status = get_training_status(item_path)
            jobs.append({
                "name": item,
                "path": item_path,
                "status": status["status"],
                "message": status["message"]
            })
    
    return jobs


def get_training_logs(output_dir: str) -> Dict[str, Any]:
    """Get training logs"""
    log_files = []
    
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith((".log", ".txt")) or file.startswith("log"):
                log_files.append(file)
    
    logs = {}
    for log_file in log_files:
        try:
            with open(os.path.join(output_dir, log_file), "r") as f:
                logs[log_file] = f.read()
        except Exception as e:
            logs[log_file] = f"Error reading log: {str(e)}"
    
    return {
        "logs": logs,
        "log_files": log_files
    }


def stop_training(output_dir: str) -> Dict[str, Any]:
    """Stop a training job (placeholder)"""
    # This is a simplified implementation
    # In a real system, you'd need to track process IDs and properly terminate them
    status_file = os.path.join(output_dir, "training_status.txt")
    
    try:
        with open(status_file, "w") as f:
            f.write("STOPPED\n")
            f.write("Training stopped via WebUI\n")
        
        return {
            "status": "stopped",
            "message": "Training marked as stopped"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to stop training: {str(e)}"
        }