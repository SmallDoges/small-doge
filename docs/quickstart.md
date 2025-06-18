# Quick Start Guide

## 🚀 Basic Usage

### 1. Model Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a SmallDoge model
model_name = "SmallDoge/Doge-60M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Generate text
prompt = "Explain the concept of machine learning:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 2. WebUI Interface

Launch the interactive web interface:

```bash
# Start WebUI (default: both backend and frontend)
small-doge-webui

# Development mode with auto-reload
small-doge-webui --dev

# Custom configuration
small-doge-webui --backend-host 127.0.0.1 --backend-port 8000 --frontend-port 7860
```

**Access URLs:**
- 🌐 **Frontend**: http://localhost:7860
- 📡 **Backend API**: http://localhost:8000  
- 📚 **API Documentation**: http://localhost:8000/docs

### 3. Jupyter Notebook Tutorial

Follow our interactive tutorials:
- [English Tutorial](../examples/notebook.ipynb)
- [中文教程](../examples/notebook_zh.ipynb)

## 📋 Available Models

### Base Models (Pre-trained)
| Model | Parameters | Speed (tokens/s on i7-11 CPU) | Use Case |
|-------|------------|---------------------------|----------|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M) | 20M | 142 | Ultra-fast prototyping |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M) | 60M | 62 | Balanced performance |
| [Doge-160M](https://huggingface.co/SmallDoge/Doge-160M) | 160M | 28 | Better reasoning |
| [Doge-320M](https://huggingface.co/SmallDoge/Doge-320M) | 320M | 16 | High performance |

### Instruction-tuned Models
| Model | Parameters | Features |
|-------|------------|----------|
| [Doge-20M-Instruct](https://huggingface.co/SmallDoge/Doge-20M-Instruct) | 20M | Chat & instruction following |
| [Doge-60M-Instruct](https://huggingface.co/SmallDoge/Doge-60M-Instruct) | 60M | Enhanced conversation |
| [Doge-160M-Instruct](https://huggingface.co/SmallDoge/Doge-160M-Instruct) | 160M | Advanced reasoning |

## 🎓 Training Your Own Model

### Quick Training Example

```python
# One-stop dataset preparation
from transformers import AutoTokenizer
from small_doge.processor.pt_datasets_process import mix_datasets_by_radio

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")

# Download, process, and mix datasets in one call
datasets_and_ratios = [
    {"fineweb-edu": 0.7},
    {"cosmopedia-v2": 0.2}, 
    {"python-edu": 0.05},
    {"finemath": 0.05},
]

mixed_dataset = mix_datasets_by_radio(
    datasets_and_ratios=datasets_and_ratios,
    total_sample_size=128000000,
    processing_class=tokenizer,
    max_length=2048,
    packing=True,
    seed=233,
    cache_dir="./cache",
)

# Save and start training
mixed_dataset.save_to_disk("./datasets/pt_dataset")
```

```bash
# Start pre-training (14 hours on RTX 4090)
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/single_gpu.yaml \
    ./src/small_doge/trainer/doge/pt.py \
    --config recipes/doge/Doge-20M/config_full.yaml
```

### Training Stages

1. **Pre-training**: Train from scratch or continue from checkpoint
   ```bash
   # Use Doge checkpoints for continued training
   # Doge-20M-checkpoint: learning_rate=8e-3
   # Doge-60M-checkpoint: learning_rate=6e-3  
   # Doge-160M-checkpoint: learning_rate=4e-3
   # Doge-320M-checkpoint: learning_rate=2e-3
   ```
   
2. **Instruction Fine-tuning**: Create chat-capable models
   ```bash
   # SFT (Supervised Fine-tuning)
   accelerate launch --config_file ../accelerate_configs/single_gpu.yaml ../../src/small_doge/trainer/doge/sft.py --config Doge-20M-Instruct/sft/config_full.yaml
   
   # DPO (Direct Preference Optimization)  
   accelerate launch --config_file ../accelerate_configs/single_gpu.yaml ../../src/small_doge/trainer/doge/dpo.py --config Doge-20M-Instruct/dpo/config_full.yaml
   ```
   
3. **Reasoning Fine-tuning**: Enhance reasoning capabilities
   ```bash
   # Distillation from teacher models
   accelerate launch --config_file ../accelerate_configs/single_gpu.yaml ../../src/small_doge/trainer/doge/sft.py --config Doge-20M-Reason/sft/config_full.yaml
   
   # GRPO (Group Relative Policy Optimization)
   accelerate launch --config_file ../accelerate_configs/single_gpu.yaml ../../src/small_doge/trainer/doge/grpo.py --config Doge-20M-Reason/grpo/config_full.yaml
   ```

📚 **Detailed guide**: [Complete Training Guide](./training.md)

## 📊 Model Evaluation

Evaluate model performance using our evaluation toolkit:

```bash
# Install evaluation toolkit
pip install lighteval

# Linux evaluation
bash ./evaluation/eval_downstream_tasks.sh

# Windows evaluation
powershell ./evaluation/eval_downstream_tasks.ps1

# Custom evaluation
export MODEL="SmallDoge/Doge-60M"
export OUTPUT_DIR="./eval_results"
bash ./evaluation/eval_downstream_tasks.sh
```

**Supported benchmarks**: MMLU, ARC, PIQA, HellaSwag, Winogrande, TriviaQA, BBH, IFEval

🔍 **Evaluation toolkit**: [Evaluation Guide](../evaluation/README.md)

## 🔧 Configuration Examples

### Low Resource Setup (4GB GPU)
```python
# Use the smallest model
model_name = "SmallDoge/Doge-20M-Instruct"

# Enable gradient checkpointing
training_args = {
    "gradient_checkpointing": True,
    "dataloader_pin_memory": False,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8
}
```

### High Performance Setup (24GB GPU)
```python
# Use larger model
model_name = "SmallDoge/Doge-320M"

# Optimize for speed
training_args = {
    "per_device_train_batch_size": 8,
    "dataloader_num_workers": 4,
    "bf16": True,
    "tf32": True
}
```

## 📱 Integration Examples

### OpenAI-compatible API
```python
import openai

# Configure client for SmallDoge WebUI
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Chat completion
response = client.chat.completions.create(
    model="SmallDoge/Doge-60M-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Streamlit App
```python
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="SmallDoge/Doge-60M-Instruct")

generator = load_model()
user_input = st.text_input("Enter your prompt:")
if user_input:
    result = generator(user_input, max_length=100)
    st.write(result[0]['generated_text'])
```

## 🎯 Next Steps

- 📖 Read detailed [Training Guide](../recipes/doge/README.md)
- 🔍 Explore [Model Documentation](./models.md)
- 🧪 Try [Advanced Examples](../examples/)
- 💬 Join our [Discord Community](https://discord.gg/P2yYH95N)
