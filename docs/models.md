# Model Documentation

## 🏗️ Architecture Overview

SmallDoge models feature innovative architectural components designed for efficiency and performance:

### Core Innovations

1. **Dynamic Mask Attention (DMA)** - Efficient attention mechanism for long sequences
2. **Cross Domain Mixture of Experts (CDMoE)** - Sparse experts with dense-to-sparse continuation training  
3. **WSD Scheduler** - Warmup-Stable-Decay for seamless checkpoint resumption

<div align="center">
    <img src="../assets/doge_architecture.png" alt="Doge Architecture" width="80%"/>
</div>

## 📊 Available Models

### 🔧 Base Models

Pre-trained foundation models for general-purpose language understanding:

| Model | Parameters | Speed (i7-11 CPU) | MMLU | HuggingFace |
|-------|------------|------------------|------|-------------|
| **Doge-20M** | 20M | 142 tok/s | 25.4 | [🤗 View Card](./Doge-20M.md) |
| **Doge-60M** | 60M | 62 tok/s | 26.4 | [🤗 View Card](./Doge-60M.md) |
| **Doge-160M** | 160M | 28 tok/s | 29.2 | [🤗 View Card](./Doge-160M.md) |
| **Doge-320M** | 320M | 16 tok/s | 33.8 | [🤗 View Card](./Doge-320M.md) |

### 💬 Instruction Models  

Chat-optimized models fine-tuned for conversations and instruction following:

| Model | Base Model | Training | HuggingFace |
|-------|------------|----------|-------------|
| **Doge-20M-Instruct** | Doge-20M | SFT + DPO | [🤗 View Card](./Doge-20M-Instruct.md) |
| **Doge-60M-Instruct** | Doge-60M | SFT + DPO | [🤗 View Card](./Doge-60M-Instruct.md) |
| **Doge-160M-Instruct** | Doge-160M | SFT + DPO | [🤗 View Card](./Doge-160M-Instruct.md) |
| **Doge-320M-Instruct** | Doge-320M | SFT + DPO | [🤗 View Card](./Doge-320M-Instruct.md) |

### 🎯 Intermediate Training Models

Partially trained models for supervised fine-tuning stages:

| Model | Training Stage | Base Model | HuggingFace |
|-------|----------------|------------|-------------|
| **Doge-20M-Instruct-SFT** | SFT Only | Doge-20M | [🤗 View Card](./Doge-20M-Instruct-SFT.md) |
| **Doge-60M-Instruct-SFT** | SFT Only | Doge-60M | [🤗 View Card](./Doge-60M-Instruct-SFT.md) |
| **Doge-160M-Instruct-SFT** | SFT Only | Doge-160M | [🤗 View Card](./Doge-160M-Instruct-SFT.md) |
| **Doge-320M-Instruct-SFT** | SFT Only | Doge-320M | [🤗 View Card](./Doge-320M-Instruct-SFT.md) |

### 🔄 Checkpoint Models

Intermediate checkpoints for continued training with stable learning rates:

| Model | Recommended LR | Scheduler | HuggingFace |
|-------|----------------|-----------|-------------|
| **Doge-20M-checkpoint** | 8e-3 | wsd_scheduler | [🤗 View Card](./Doge-20M-checkpoint.md) |
| **Doge-60M-checkpoint** | 6e-3 | wsd_scheduler | [🤗 View Card](./Doge-60M-checkpoint.md) |
| **Doge-160M-checkpoint** | 4e-3 | wsd_scheduler | [🤗 View Card](./Doge-160M-checkpoint.md) |
| **Doge-320M-checkpoint** | 2e-3 | wsd_scheduler | [🤗 View Card](./Doge-320M-checkpoint.md) |

### 🧠 Reasoning Models

Advanced models enhanced with reasoning capabilities through knowledge distillation:

| Model | Training Method | Capabilities | HuggingFace |
|-------|-----------------|--------------|-------------|
| **Doge-160M-Reason-Distill** | Knowledge Distillation + GRPO | Chain-of-thought reasoning | [🤗 View Card](./Doge-160M-Reason-Distill.md) |


## ⚡ Quick Start

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load any model (example: instruction model)
model_name = "SmallDoge/Doge-60M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Generate text
prompt = "Explain machine learning in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

For detailed usage examples, see individual model cards above.

## 🎯 Model Selection Guide

### Choose by Use Case

- **🔬 Research & Experimentation**: Start with Doge-20M for fast iteration
- **💻 Development & Prototyping**: Use Doge-60M for balanced performance  
- **🎯 Production Applications**: Deploy Doge-160M or Doge-320M for best quality
- **💬 Chat Applications**: Use `-Instruct` variants for conversation
- **🧠 Reasoning Tasks**: Try Doge-160M-Reason-Distill for complex problems
- **📚 Continued Training**: Use `-checkpoint` models with specified learning rates

### Choose by Resources

- **CPU-only**: Doge-20M (142 tok/s) or Doge-60M (62 tok/s)
- **Mobile/Edge**: Doge-20M with quantization
- **GPU Available**: Any model, Doge-320M recommended for best results
- **Memory Constrained**: Doge-20M (0.5GB) or Doge-60M (1.2GB)

## 📚 Documentation

- **[Training Guide](./training.md)** - Complete training pipeline
- **[Quick Start](./quickstart.md)** - Get started in 5 minutes
- **[Installation](./installation.md)** - Setup instructions
- **[WebUI Guide](./webui.md)** - Web interface usage
