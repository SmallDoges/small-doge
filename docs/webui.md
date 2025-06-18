# WebUI Documentation

## 🌐 Overview

SmallDoge WebUI provides an intuitive interface for interacting with SmallDoge models, featuring real-time chat, model management, and OpenAI-compatible API endpoints.

## 🚀 Quick Start

### Installation

```bash
# Install with WebUI support
pip install -e '.[webui]'

# Or install dependencies separately
pip install -e '.[webui-backend]'  # Backend only
pip install -e '.[webui-frontend]' # Frontend only
```

### Launch WebUI

```bash
# Start complete WebUI (default)
small-doge-webui

# Development mode with auto-reload
small-doge-webui --dev

# Start only backend API
small-doge-webui --backend-only

# Start only frontend
small-doge-webui --frontend-only

# Custom configuration
small-doge-webui --backend-host 0.0.0.0 --backend-port 8000 --frontend-port 7860
```

## 🌍 Access Points

### Frontend Interface
- **URL**: http://localhost:7860
- **Features**: Chat interface, model selection, settings
- **Technology**: Enhanced Gradio with custom styling

### Backend API
- **URL**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (Swagger UI)
- **Redoc**: http://localhost:8000/redoc
- **Compatibility**: OpenAI API standard

## 💬 Chat Interface Features

### Core Features
- ✨ **Zero Authentication**: Start chatting immediately
- 💬 **Real-time Streaming**: Live token-by-token responses
- 📱 **Responsive Design**: Works on desktop and mobile
- 🎨 **Modern UI**: Clean, intuitive interface
- 🔄 **Model Switching**: Change models without restart

### Advanced Features
- 🧵 **Conversation History**: Persistent chat sessions
- ⚙️ **Parameter Control**: Temperature, max tokens, top-p
- 📊 **Performance Metrics**: Speed and token statistics
- 💾 **Export/Import**: Save and load conversations
- 🎯 **System Prompts**: Custom instructions for models

## 🔧 Configuration

### Environment Variables

```bash
# Backend configuration
export SMALLDOGE_HOST=0.0.0.0
export SMALLDOGE_PORT=8000
export SMALLDOGE_MODEL=SmallDoge/Doge-60M-Instruct

# Frontend configuration
export GRADIO_SERVER_PORT=7860
export GRADIO_SERVER_NAME=0.0.0.0

# Model configuration
export TRUST_REMOTE_CODE=true
export DEVICE=auto
export TORCH_DTYPE=float16
```

### Configuration File

Create `config.yaml`:

```yaml
# Model settings
model:
  name: "SmallDoge/Doge-60M-Instruct"
  device: "auto"
  torch_dtype: "float16"
  trust_remote_code: true

# Server settings
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1

# Generation settings
generation:
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 50

# WebUI settings
webui:
  port: 7860
  theme: "soft"
  enable_chat_history: true
```

## 📡 API Usage

### OpenAI-Compatible Endpoints

#### Chat Completions

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required
)

response = client.chat.completions.create(
    model="SmallDoge/Doge-60M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain machine learning"}
    ],
    max_tokens=200,
    temperature=0.7,
    stream=True  # Enable streaming
)

# Handle streaming response
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

#### Completions (Legacy)

```python
response = client.completions.create(
    model="SmallDoge/Doge-60M",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].text)
```

### Direct API Usage

```python
import requests

# Chat completion request
payload = {
    "model": "SmallDoge/Doge-60M-Instruct",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": False
}

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json=payload
)

print(response.json())
```

### Streaming API

```python
import requests
import json

def stream_chat(messages):
    payload = {
        "model": "SmallDoge/Doge-60M-Instruct",
        "messages": messages,
        "stream": True
    }
    
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data != '[DONE]':
                    chunk = json.loads(data)
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        yield content

# Usage
messages = [{"role": "user", "content": "Write a poem"}]
for token in stream_chat(messages):
    print(token, end="", flush=True)
```

## 🎛️ Model Management

### Available Models

The WebUI automatically detects and lists available SmallDoge models:

- Doge-20M / Doge-20M-Instruct
- Doge-60M / Doge-60M-Instruct  
- Doge-160M / Doge-160M-Instruct
- Doge-320M / Doge-320M-Instruct
- Custom trained models

### Model Loading

```python
# Load model via API
import requests

response = requests.post(
    "http://localhost:8000/v1/models/load",
    json={"model_name": "SmallDoge/Doge-160M-Instruct"}
)

print(response.json())
```

### Model Information

```python
# Get model details
response = requests.get("http://localhost:8000/v1/models")
models = response.json()

for model in models["data"]:
    print(f"Model: {model['id']}")
    print(f"Created: {model['created']}")
    print(f"Owned by: {model['owned_by']}")
```

## 🔒 Security & Deployment

### Production Deployment

```bash
# Use production WSGI server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker small_doge.webui.backend:app \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --worker-connections 1000
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -e '.[webui]'

EXPOSE 8000 7860

CMD ["small-doge-webui", "--backend-host", "0.0.0.0", "--frontend-host", "0.0.0.0"]
```

```bash
# Build and run
docker build -t smalldoge-webui .
docker run -p 8000:8000 -p 7860:7860 smalldoge-webui
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # API Backend
    location /v1/ {
        proxy_pass http://localhost:8000/v1/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Support for streaming
        proxy_buffering off;
        proxy_read_timeout 300;
    }
}
```

## 🔧 Customization

### Custom Themes

```python
import gradio as gr

# Custom CSS
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.chat-message {
    border-radius: 15px;
    padding: 10px;
    margin: 5px 0;
}
"""

# Apply custom theme
webui = gr.Interface(
    fn=chat_function,
    css=custom_css,
    theme=gr.themes.Soft()
)
```

### Custom Components

```python
from small_doge.webui import WebUIBuilder

builder = WebUIBuilder()

# Add custom tab
builder.add_tab(
    name="Model Analytics",
    content=create_analytics_interface()
)

# Add custom endpoint
builder.add_endpoint(
    path="/custom/analyze",
    method="POST",
    handler=custom_analysis_handler
)

webui = builder.build()
```

## 📊 Monitoring & Analytics

### Built-in Metrics

- Request count and rate
- Response time statistics
- Token generation speed
- Memory usage
- GPU utilization (if available)

### Prometheus Integration

```python
from prometheus_client import Counter, Histogram, generate_latest

# Custom metrics
request_count = Counter('smalldoge_requests_total', 'Total requests')
response_time = Histogram('smalldoge_response_time_seconds', 'Response time')

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check model availability
   curl http://localhost:8000/v1/models
   
   # Verify trust_remote_code setting
   export TRUST_REMOTE_CODE=true
   ```

2. **Memory Issues**
   ```python
   # Use smaller models or quantization
   model_name = "SmallDoge/Doge-20M-Instruct"  # Smaller model
   torch_dtype = torch.float16  # Half precision
   ```

3. **Slow Response**
   ```bash
   # Check system resources
   nvidia-smi  # GPU usage
   htop        # CPU/Memory usage
   
   # Optimize settings
   export TORCH_NUM_THREADS=4
   export OMP_NUM_THREADS=4
   ```

### Debug Mode

```bash
# Enable debug logging
small-doge-webui --debug

# Or set environment variable
export SMALLDOGE_DEBUG=true
```

## 📚 Integration Examples

### Streamlit Integration

```python
import streamlit as st
import requests

st.title("SmallDoge Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from SmallDoge API
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "SmallDoge/Doge-60M-Instruct",
            "messages": st.session_state.messages
        }
    )
    
    assistant_response = response.json()["choices"][0]["message"]["content"]
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
```

### Discord Bot Integration

```python
import discord
from discord.ext import commands
import requests

bot = commands.Bot(command_prefix='!')

@bot.command(name='ask')
async def ask_smalldoge(ctx, *, question):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "SmallDoge/Doge-60M-Instruct",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 200
        }
    )
    
    answer = response.json()["choices"][0]["message"]["content"]
    await ctx.send(answer)

bot.run('YOUR_BOT_TOKEN')
```

## 📋 API Reference

### Complete Endpoint List

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/models/{model}` | GET | Get model details |
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/completions` | POST | Text completion |
| `/v1/embeddings` | POST | Get embeddings |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

For detailed API documentation, visit `http://localhost:8000/docs` when the server is running.
