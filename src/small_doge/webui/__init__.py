# coding=utf-8
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
SmallDoge WebUI - A model inference WebUI with Gradio frontend and FastAPI backend
"""

# Version information
__version__ = "0.1.0"

# Import webui components
try:
    from .frontend.app import SmallDogeWebUI
    from .run import main as run_webui
    
    __all__ = [
        "SmallDogeWebUI",
        "run_webui",
        "__version__",
    ]
except ImportError:
    # Handle case where webui dependencies are not installed
    __all__ = ["__version__"]


def launch_webui():
    """
    Command-line entry point for launching the WebUI
    Supports all command line arguments from run.py
    
    Usage:
        small-doge-webui
        small-doge-webui --dev
        small-doge-webui --backend-only
        small-doge-webui --frontend-only
        small-doge-webui --management  # New management interface
    """
    import sys
    import os
    from pathlib import Path
    
    # Check if --management flag is provided
    if "--management" in sys.argv:
        sys.argv.remove("--management")
        return launch_management_interface()
    
    # Get the webui directory
    webui_dir = Path(__file__).parent
    
    # Add webui directory to Python path for proper imports
    webui_str = str(webui_dir)
    if webui_str not in sys.path:
        sys.path.insert(0, webui_str)
    
    # Change to webui directory to ensure proper file operations
    original_cwd = os.getcwd()
    os.chdir(webui_dir)
    
    try:
        from .run import main
        return main()
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        # Remove from path if we added it
        if webui_str in sys.path:
            sys.path.remove(webui_str)


def launch_management_interface():
    """
    Launch the dataset and training management interface
    """
    import sys
    import os
    from pathlib import Path
    
    # Get the webui directory
    webui_dir = Path(__file__).parent
    
    # Add webui directory to Python path for proper imports
    webui_str = str(webui_dir)
    if webui_str not in sys.path:
        sys.path.insert(0, webui_str)
    
    # Change to webui directory to ensure proper file operations
    original_cwd = os.getcwd()
    os.chdir(webui_dir)
    
    try:
        from .frontend.management_app import main
        return main()
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        # Remove from path if we added it
        if webui_str in sys.path:
            sys.path.remove(webui_str)


def launch_webui_programmatic(*args, **kwargs):
    """
    Programmatic function to launch the WebUI with custom arguments
    
    Usage:
        from small_doge.webui import launch_webui_programmatic
        launch_webui_programmatic()
    """
    from .run import main
    return main(*args, **kwargs)


# Add functions to exports
__all__.extend(["launch_webui", "launch_webui_programmatic", "launch_management_interface"]) 