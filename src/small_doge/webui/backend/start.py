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

#!/usr/bin/env python3
"""
Startup script for SmallDoge WebUI Backend
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

try:
    import uvicorn
    from small_doge.webui.backend.smalldoge_webui.main import app
    from small_doge.webui.backend.smalldoge_webui.env import HOST, PORT, ENV, log
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install the required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SmallDoge WebUI Backend Server")
    
    parser.add_argument(
        "--host",
        type=str,
        default=HOST,
        help=f"Host to bind to (default: {HOST})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"Port to bind to (default: {PORT})"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        default=ENV == "dev",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Log level (default: info)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--access-log",
        action="store_true",
        default=False,
        help="Enable access logging"
    )
    
    return parser.parse_args()


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "sqlalchemy",
        "torch",
        "transformers",
        "bcrypt",
        "jose"  # python-jose imports as 'jose'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them with:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def setup_logging(log_level: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Print startup information
    log.info("=" * 50)
    log.info("SmallDoge WebUI Backend Server")
    log.info("=" * 50)
    log.info(f"Host: {args.host}")
    log.info(f"Port: {args.port}")
    log.info(f"Environment: {ENV}")
    log.info(f"Reload: {args.reload}")
    log.info(f"Workers: {args.workers}")
    log.info(f"Log Level: {args.log_level}")
    log.info("=" * 50)
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "smalldoge_webui.main:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "access_log": args.access_log,
    }
    
    if args.reload:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = [str(backend_dir)]
        # Exclude model cache directory from file watching
        uvicorn_config["reload_excludes"] = [
            str(backend_dir / "smalldoge_webui" / "data" / "cache"),
            "*.safetensors",
            "*.bin",
            "*.json",
            "*.py"  # Exclude downloaded model files
        ]
    else:
        uvicorn_config["workers"] = args.workers
    
    try:
        # Start the server
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        log.info("Server stopped by user")
    except Exception as e:
        log.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
