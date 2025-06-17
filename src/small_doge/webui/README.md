# 🐕 SmallDoge WebUI

**Open Source Feature Sharing Platform** - A model inference WebUI designed for collaborative AI experimentation and feature sharing without barriers.

## 🎯 Project Goal

SmallDoge WebUI is built for **open source feature sharing** - enabling researchers, developers, and AI enthusiasts to:
- Share AI model capabilities instantly without setup friction
- Collaborate on model experiments in real-time
- Demonstrate model features to the community
- Provide immediate access to AI capabilities for everyone

**No login, no barriers, just pure AI interaction!**

## ✨ Features

### 🚀 **Enhanced Frontend Experience**
- **💬 Real-time Streaming**: Live chat responses with token-by-token streaming
- **📱 Modern UI/UX**: Enhanced Gradio interface matching open-webui design patterns
- **💾 Chat Persistence**: Automatic chat history saving and session management
- **📋 Multi-Session Support**: Create, switch, and manage multiple chat sessions
- **📤 Export/Import**: Export chat sessions for sharing and backup
- **🔄 Auto-Refresh**: Real-time backend status monitoring
- **⚡ Instant Access**: Zero authentication - start chatting immediately

### 🔧 **Robust Backend Architecture**
- **🌐 OpenAI-Compatible API**: Full compatibility with OpenAI API standards
- **📡 Enhanced Endpoints**: Extended API with model management and health checks
- **🔄 Dynamic Model Loading**: Load/unload models on demand with status monitoring
- **📊 Usage Statistics**: Track model usage and performance metrics
- **🛡️ Error Handling**: Comprehensive error handling and recovery mechanisms
- **🔗 Streaming Support**: Optimized server-sent events for real-time responses

### 🤖 **SmallDoge Optimization**
- **🐕 SmallDoge Models**: Built specifically for SmallDoge models with `trust_remote_code=True`
- **🎯 Model Capabilities**: Automatic detection and display of model capabilities
- **📏 Context Management**: Intelligent context length handling and token management
- **⚙️ Parameter Control**: Fine-grained control over temperature, top-p, and max tokens

## 🏗️ Architecture

**Open Source Sharing First** - Designed for immediate access and collaboration:

```
smalldoge-webui/
├── backend/                     # FastAPI backend (no auth required)
│   ├── smalldoge_webui/         # Main package
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration management
│   │   ├── models/              # Data models (chats, models)
│   │   ├── routers/             # API routes (open access)
│   │   ├── utils/               # Utility functions
│   │   └── internal/            # Internal services
│   ├── requirements.txt         # Backend dependencies
│   └── start.py                 # Backend startup script
├── frontend/                    # Gradio frontend (instant access)
│   ├── app.py                   # Main Gradio application
│   └── requirements.txt         # Frontend dependencies
└── README.md
```

**Key Design Principles:**
- 🔓 **Open by Default**: No authentication barriers
- 🤝 **Collaboration Ready**: Built for sharing and experimentation
- ⚡ **Fast Setup**: Get running in minutes, not hours
- 🔧 **Developer Friendly**: Clean APIs and extensible architecture

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- Git

### Installation

1. **Navigate to repository root**:
   ```bash
   cd /path/to/small-doge
   ```

2. **Install WebUI**:
   ```bash
   pip install -e '.[webui]'
   ```

### Running the Application

1. **Launch WebUI**:
   ```bash
   # Start WebUI (both backend and frontend)
   small-doge-webui

   # Development mode with auto-reload
   small-doge-webui --dev

   # Start only backend
   small-doge-webui --backend-only

   # Start only frontend
   small-doge-webui --frontend-only
   ```

2. **Start sharing AI features**:
   - Open your browser and go to `http://localhost:7860`
   - **No setup, no login, no barriers** - start chatting immediately!
   - Share the URL with others for instant collaboration

## ⚙️ Configuration

### Environment Variables (Optional)

The WebUI works out-of-the-box with sensible defaults. For customization, create a `.env` file in the backend directory:

```env
# Database (SQLite by default)
DATABASE_URL=sqlite:///./data/smalldoge_webui.db

# Models
DEFAULT_MODELS=SmallDoge/Doge-160M
MODEL_CACHE_DIR=./data/models

# Server
HOST=0.0.0.0
PORT=8000
ENV=dev

# CORS (enabled for open sharing)
ENABLE_CORS=true
CORS_ALLOW_ORIGIN=*

# Model Inference
MAX_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
DEVICE=auto
TORCH_DTYPE=auto
```

**Note**: Authentication is permanently disabled to support the open source sharing mission.

### Model Configuration

The application supports various SmallDoge models:
- `SmallDoge/Doge-160M` (default)
- `SmallDoge/Doge-320M`

Models are automatically loaded with `trust_remote_code=True` for compatibility.

## 📡 API Documentation

**Open Access APIs** - No authentication required for any endpoint!

After starting the WebUI, you can view complete API documentation in your browser:

- **OpenAI-Compatible API**: http://localhost:8000/docs 
- **Interactive API Docs**: http://localhost:8000/redoc
- **API Overview**: http://localhost:8000/

Main API Categories:
- 🤖 **OpenAI-Compatible Endpoints** - Standard Chat Completions API
- 🔧 **Model Management** - Dynamic model loading/unloading
- 💬 **Chat Management** - Session creation and history management
- 🔍 **System Monitoring** - Health checks and status monitoring

## Development

### Backend Development

1. **Install development dependencies**:
   ```bash
   # Development dependencies are included in the main package
   pip install -e '.[dev,webui]'
   ```

2. **Run tests**:
   ```bash
   pytest
   ```

3. **Format code**:
   ```bash
   black .
   isort .
   ```

4. **Start with auto-reload**:
   ```bash
   python start.py --reload
   ```

### Frontend Development

The Gradio frontend automatically reloads when files change during development.

### Adding New Models

1. Add model ID to `MODEL_CONFIG.SMALLDOGE_MODELS` in `constants.py`
2. Update model-specific configurations if needed
3. Test model loading and inference


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing to Open Source AI

SmallDoge WebUI embodies the spirit of open collaboration. We encourage:

- **Feature Sharing**: Deploy your instance and share cool AI capabilities
- **Model Experiments**: Try new models and share your findings
- **Community Building**: Help others discover and use AI tools
- **Open Development**: Contribute code, ideas, and feedback

### Ways to Contribute

1. **Share Your Instance**: Deploy and share your WebUI with the community
2. **Add Models**: Integrate new models and share configurations
3. **Improve Features**: Submit PRs for new functionality
4. **Report Issues**: Help us improve the platform
5. **Spread the Word**: Share the project with other AI enthusiasts

## 🙏 Acknowledgments

- Inspired by [open-webui](https://github.com/open-webui/open-webui) architecture patterns
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [Gradio](https://gradio.app/)
- Powered by [Transformers](https://huggingface.co/transformers/) and [PyTorch](https://pytorch.org/)
- Made possible by the open source AI community

---

**🎯 Mission**: Making AI accessible to everyone through open source feature sharing.

**🚀 Vision**: A world where AI capabilities are shared freely and collaboratively.
