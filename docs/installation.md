# Installation Guide

## System Requirements

### For Training and Fine-tuning
- **OS**: Windows or Linux
- **GPU**: NVIDIA GPU with CUDA support
- **Python**: 3.10+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+

### For Inference Only
- **OS**: Windows, Linux, or macOS
- **Python**: 3.8+
- **Hardware**: CPU or GPU

## Installation Methods

### Method 1: Standard Installation

```bash
git clone https://github.com/SmallDoges/small-doge.git
cd small-doge
pip install -e .
```

### Method 2: Docker Installation (Recommended for Training)

```bash
docker pull nvcr.io/nvidia/pytorch:24.12-py3
docker run --privileged --gpus all -it --name SmallDoge \
  --shm-size=32g -p 8888:8888 -p 6006:6006 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v <your_code_path>:/workspace \
  -v <your_datasets_path>:/workspace/datasets \
  nvcr.io/nvidia/pytorch:24.12-py3
```

### Method 3: WebUI Installation

For interactive model usage with web interface:

```bash
# Install with WebUI support
pip install -e '.[webui]'

# Or install only backend dependencies
pip install -e '.[webui-backend]'

# Or install only frontend dependencies  
pip install -e '.[webui-frontend]'
```

## Core Dependencies

The installation will automatically install these core packages:

- `transformers`: Core framework for model operations
- `datasets`: Dataset handling and processing
- `sentencepiece`: Tokenization support
- `boto3`: AWS S3 dataset downloads
- `accelerate`: Distributed training support
- `trl`: Reinforcement learning fine-tuning
- `torch`: Deep learning framework

## Verification

Verify your installation:

```python
import small_doge
from transformers import AutoTokenizer, AutoModelForCausalLM

# Test model loading
model_name = "SmallDoge/Doge-20M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

print("✅ Installation successful!")
```

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Permission Errors**
   ```bash
   pip install -e . --user
   ```

3. **Memory Issues**
   - Ensure adequate RAM (8GB+ recommended)
   - For training, 16GB+ GPU memory recommended

### Getting Help

- 📖 Check our [documentation](../docs/)
- 💬 Join our [Discord community](https://discord.gg/P2yYH95N)
- 🐛 Report issues on [GitHub](https://github.com/SmallDoges/small-doge/issues)
