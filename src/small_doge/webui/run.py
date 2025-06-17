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
Startup script for SmallDoge WebUI
Starts both backend and frontend servers
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SmallDoge WebUI Launcher")
    
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Start only the backend server"
    )
    
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Start only the frontend server"
    )
    
    parser.add_argument(
        "--backend-host",
        type=str,
        default="localhost",
        help="Backend host (default: localhost)"
    )
    
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="Backend port (default: 8000)"
    )
    
    parser.add_argument(
        "--frontend-host",
        type=str,
        default="localhost",
        help="Frontend host (default: localhost)"
    )
    
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=7860,
        help="Frontend port (default: 7860)"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode with auto-reload"
    )
    
    return parser.parse_args()


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    # Check backend dependencies
    backend_packages = ["fastapi", "uvicorn", "transformers", "torch", "jose"]
    missing_backend = []

    for package in backend_packages:
        try:
            __import__(package)
        except ImportError:
            missing_backend.append(package)

    if missing_backend:
        print(f"✗ Missing backend dependencies: {', '.join(missing_backend)}")
        print("Please install WebUI dependencies:")
        print("pip install -e '.[webui]' or pip install -e '.[webui-backend]'")
        return False
    else:
        print("✓ Backend dependencies found")

    # Check frontend dependencies
    frontend_packages = ["gradio", "requests"]
    missing_frontend = []

    for package in frontend_packages:
        try:
            __import__(package)
        except ImportError:
            missing_frontend.append(package)

    if missing_frontend:
        print(f"✗ Missing frontend dependencies: {', '.join(missing_frontend)}")
        print("Please install WebUI dependencies:")
        print("pip install -e '.[webui]' or pip install -e '.[webui-frontend]'")
        return False
    else:
        print("✓ Frontend dependencies found")
    
    return True


def start_backend(host: str, port: int, dev: bool = False):
    """Start the backend server"""
    print(f"Starting backend server on {host}:{port}")
    
    backend_dir = Path(__file__).parent / "backend"
    cmd = [
        sys.executable, "start.py",
        "--host", host,
        "--port", str(port)
    ]
    
    if dev:
        cmd.append("--reload")
    
    return subprocess.Popen(
        cmd,
        cwd=backend_dir,
        env={**os.environ, "PYTHONPATH": str(backend_dir)}
    )


def start_frontend(host: str, port: int, backend_url: str):
    """Start the frontend server"""
    print(f"Starting frontend server on {host}:{port}")
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    # Set environment variables for frontend
    env = os.environ.copy()
    env["BACKEND_URL"] = backend_url
    env["GRADIO_SERVER_NAME"] = host
    env["GRADIO_SERVER_PORT"] = str(port)
    
    return subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=frontend_dir,
        env=env
    )


def wait_for_backend(host: str, port: int, timeout: int = 30):
    """Wait for backend to be ready"""
    import requests
    
    url = f"http://{host}:{port}/health"
    print(f"Waiting for backend to be ready at {url}")
    
    for i in range(timeout):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print("✓ Backend is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        if i % 5 == 0:
            print(f"Still waiting for backend... ({i}/{timeout}s)")
    
    print("✗ Backend failed to start within timeout")
    return False


def main():
    """Main entry point"""
    args = parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("=" * 60)
    print("🐕 SmallDoge WebUI")
    print("=" * 60)
    
    processes = []
    
    try:
        # Start backend
        if not args.frontend_only:
            backend_process = start_backend(
                args.backend_host, 
                args.backend_port, 
                args.dev
            )
            processes.append(("Backend", backend_process))
            
            # Wait for backend to be ready before starting frontend
            if not args.backend_only:
                backend_url = f"http://{args.backend_host}:{args.backend_port}"
                if not wait_for_backend(args.backend_host, args.backend_port):
                    print("Failed to start backend, exiting...")
                    backend_process.terminate()
                    sys.exit(1)
        
        # Start frontend
        if not args.backend_only:
            backend_url = f"http://{args.backend_host}:{args.backend_port}"
            frontend_process = start_frontend(
                args.frontend_host,
                args.frontend_port,
                backend_url
            )
            processes.append(("Frontend", frontend_process))
        
        # Print access information
        print("\n" + "=" * 60)
        print("🚀 SmallDoge WebUI is running!")
        print("=" * 60)
        
        if not args.frontend_only:
            print(f"📡 Backend API: http://{args.backend_host}:{args.backend_port}")
            print(f"📚 API Docs: http://{args.backend_host}:{args.backend_port}/docs")
        
        if not args.backend_only:
            print(f"🌐 Frontend: http://{args.frontend_host}:{args.frontend_port}")
        
        print("\nPress Ctrl+C to stop all servers")
        print("=" * 60)
        
        # Wait for processes
        while True:
            time.sleep(1)
            
            # Check if any process has died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n{name} process has stopped unexpectedly")
                    raise KeyboardInterrupt
    
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        
        # Terminate all processes
        for name, process in processes:
            print(f"Stopping {name}...")
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing {name}...")
                process.kill()
        
        print("All servers stopped.")
    
    except Exception as e:
        print(f"Error: {e}")
        
        # Clean up processes
        for name, process in processes:
            process.terminate()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
