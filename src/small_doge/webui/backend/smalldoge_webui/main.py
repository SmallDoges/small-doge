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
Main FastAPI application for SmallDoge WebUI
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from small_doge.webui.backend.smalldoge_webui.env import (
    ENV,
    HOST,
    PORT,
    ENABLE_CORS,
    CORS_ALLOW_ORIGIN,
    WEBUI_NAME,
    VERSION,
    log
)
from small_doge.webui.backend.smalldoge_webui.constants import ERROR_MESSAGES
from small_doge.webui.backend.smalldoge_webui.internal.db import create_tables, check_database_connection
from small_doge.webui.backend.smalldoge_webui.utils.models import load_default_model
from small_doge.webui.backend.smalldoge_webui.utils.task_manager import task_manager
from small_doge.webui.backend.smalldoge_webui.routers import (
    chats_router,
    models_router,
    openai_router,
    huggingface_router
)


####################
# Application Lifecycle
####################

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    log.info(f"Starting {WEBUI_NAME} v{VERSION}")
    
    try:
        # Check database connection
        if not check_database_connection():
            log.error("Failed to connect to database")
            raise Exception("Database connection failed")
        
        # Create database tables
        create_tables()
        log.info("Database initialized successfully")
        
        # Start task manager
        await task_manager.start()
        log.info("Task manager started successfully")
        
        # Load default model
        try:
            await load_default_model()
            log.info("Default model loaded successfully")
        except Exception as e:
            log.warning(f"Failed to load default model: {e}")
        
        log.info(f"{WEBUI_NAME} started successfully")
        
    except Exception as e:
        log.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    log.info(f"Shutting down {WEBUI_NAME}")
    
    # Stop task manager
    await task_manager.stop()
    log.info("Task manager stopped")


####################
# FastAPI Application
####################

app = FastAPI(
    title=WEBUI_NAME,
    description="SmallDoge WebUI - A model inference WebUI with Gradio frontend and open-webui-compatible backend",
    version=VERSION,
    docs_url="/docs" if ENV == "dev" else None,
    openapi_url="/openapi.json" if ENV == "dev" else None,
    redoc_url="/redoc" if ENV == "dev" else None,
    lifespan=lifespan
)


####################
# Middleware
####################

# CORS Middleware
if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[CORS_ALLOW_ORIGIN] if CORS_ALLOW_ORIGIN != "*" else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Trusted Host Middleware (for production)
if ENV == "prod":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure this properly in production
    )


####################
# Request Logging Middleware
####################

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    start_time = time.time()
    
    # Skip logging for health check and static files
    if request.url.path in ["/health", "/favicon.ico"]:
        response = await call_next(request)
        return response
    
    # Log request
    log.info(f"{request.method} {request.url.path} - {request.client.host if request.client else 'unknown'}")
    
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        log.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
        
        return response
    
    except Exception as e:
        process_time = time.time() - start_time
        log.error(f"{request.method} {request.url.path} - ERROR: {str(e)} - {process_time:.3f}s")
        raise


####################
# Exception Handlers
####################

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "http_error",
                "code": exc.status_code
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    log.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": ERROR_MESSAGES.DEFAULT(str(exc)),
                "type": "internal_error",
                "code": 500
            }
        }
    )


####################
# Health Check
####################

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": WEBUI_NAME,
        "version": VERSION,
        "timestamp": int(time.time())
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {WEBUI_NAME}",
        "version": VERSION,
        "docs": "/docs" if ENV == "dev" else None,
        "api": {
            "models": "/api/v1/models",
            "chats": "/api/v1/chats",
            "openai": "/openai"
        }
    }


####################
# API Routes
####################

# Authentication and user management removed for open source sharing

# Chat management routes
app.include_router(
    chats_router,
    prefix="/api/v1/chats",
    tags=["chats"]
)

# Model management routes
app.include_router(
    models_router,
    prefix="/api/v1/models",
    tags=["models"]
)

# OpenAI-compatible routes
app.include_router(
    openai_router,
    prefix="/openai",
    tags=["openai"]
)

# HuggingFace Hub integration routes
app.include_router(
    huggingface_router,
    prefix="/api/v1/huggingface",
    tags=["huggingface"]
)


####################
# Static Files (for future frontend integration)
####################

# Mount static files directory if it exists
try:
    from pathlib import Path
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        log.info("Static files mounted at /static")
except Exception as e:
    log.warning(f"Failed to mount static files: {e}")


####################
# Development Server
####################

if __name__ == "__main__":
    import uvicorn
    
    log.info(f"Starting development server on {HOST}:{PORT}")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=ENV == "dev",
        log_level="info" if ENV == "dev" else "warning"
    )
