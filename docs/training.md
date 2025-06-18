# Training Guide

## 🎯 Overview

SmallDoge supports a complete three-stage training pipeline:

1. **Pre-training** → Base models (Doge-Base)
2. **Instruction Fine-tuning** → Chat models (Doge-Instruct) 
3. **Reasoning Fine-tuning** → Reasoning models (Doge-Reason)

## 📋 Prerequisites

- NVIDIA GPU (4GB+ for 20M models, 16GB+ for 320M models)
- Python 3.10+ with PyTorch 2.0+
- See [Installation Guide](./installation.md) for detailed setup

## 🚀 Stage 1: Pre-training (Base Models)

We provide Doge checkpoints that can be further pre-trained on new datasets. See [Doge-checkpoint collection](https://huggingface.co/collections/SmallDoge/doge-checkpoint-679ce95dae1d498e3ad35068) for more information.

### 1. Prepare Dataset (One-Stop Solution)

Use the advanced dataset processor that handles download, processing, and mixing in one function:

```python
from transformers import AutoTokenizer
from small_doge.processor.pt_datasets_process import mix_datasets_by_radio

# Define datasets and mixing ratios (fineweb-edu:cosmopedia-v2:python-edu:finemath = 7:2:0.5:0.5)
datasets_and_ratios = [
    {"fineweb-edu": 0.7},
    {"cosmopedia-v2": 0.2}, 
    {"python-edu": 0.05},
    {"finemath": 0.05},
]

# Load Doge tokenizer (vocab_size=32768)
tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")

# One function to download, process, and mix datasets
mixed_dataset = mix_datasets_by_radio(
    datasets_and_ratios=datasets_and_ratios,
    total_sample_size=128000000,  # 128M samples for training
    dataset_text_field="text",
    processing_class=tokenizer,
    max_length=2048,
    packing=True,
    formatting_func=None,  # Custom formatting if needed
    dataset_num_proc=16,
    seed=233,
    cache_dir="./cache",
)

# Save processed dataset
mixed_dataset.save_to_disk("./datasets/pt_dataset")
```

**Benefits:**
- ✅ **All-in-one**: Download, tokenize, and mix datasets in one call
- ✅ **Flexible ratios**: Easy to adjust dataset mixing proportions
- ✅ **Memory efficient**: Processes datasets in streaming mode
- ✅ **Reproducible**: Fixed seed for consistent results
- ✅ **Scalable**: Parallel processing with configurable workers

### 2. Model Configuration

| Model | Params | n_layers | d_model | d_ff | n_heads | kv_heads |
|-------|--------|----------|---------|------|---------|----------|
| **Doge-20M** | 13M | 8 | 256 | 512 | 2 | 1 |
| **Doge-60M** | 54M | 16 | 512 | 1024 | 4 | 2 |
| **Doge-160M** | 152M | 24 | 768 | 1536 | 6 | 3 |
| **Doge-320M** | 335M | 32 | 1024 | 2048 | 8 | 4 |

### 3. Training Hyperparameters

| Model | Tokens | Steps | Accumulate | Learning Rate | Scheduler | GPU Hours (RTX 4090) |
|-------|--------|-------|------------|---------------|-----------|----------------------|
| **Doge-20M** | 4B | 8,000 | 256 | 8e-3 | warmup_stable_decay | 14 |
| **Doge-60M** | 16B | 16,000 | 512 | 6e-3 | warmup_stable_decay | 128 |
| **Doge-160M** | 32B | 24,000 | 768 | 4e-3 | warmup_stable_decay | 522 |
| **Doge-320M** | 64B | 32,000 | 1024 | 2e-3 | warmup_stable_decay | 1856 |

### 4. Start Pre-training

```bash
# Single GPU training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/single_gpu.yaml \
    ./src/small_doge/trainer/doge/pt.py \
    --config recipes/doge/Doge-20M/config_full.yaml
```

**Multi-GPU Options:**
- `single_gpu.yaml` - Single GPU training
- `ddp.yaml` - Distributed Data Parallel
- `deepspeed_zero2.yaml` - DeepSpeed ZeRO-2
- `deepspeed_zero3.yaml` - DeepSpeed ZeRO-3

## 🎯 Stage 2: Instruction Fine-tuning (Chat Models)

We provide base Doge models for instruction fine-tuning. See [Doge-SLM collection](https://huggingface.co/collections/SmallDoge/doge-slm-679cc991f027c4a3abbded4a) for available models.

### 1. Prepare Dataset (One-Stop Solution)

Use the advanced dataset processor for instruction fine-tuning:

```python
from transformers import AutoTokenizer
from small_doge.processor.ft_datasets_process import mix_datasets_by_radio

# Define instruction datasets and ratios
datasets_and_ratios = [
    {"smoltalk": 0.8},                    # Conversation dataset for SFT
    {"ultrafeedback_binarized": 0.2},     # Preference dataset for DPO
]

# Load Doge tokenizer
tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")

# One function to download, process, and mix instruction datasets
mixed_dataset = mix_datasets_by_radio(
    datasets_and_ratios=datasets_and_ratios,
    total_sample_size=1000000,  # 1M samples for instruction tuning
    dataset_text_field="text",  # or conversation field
    processing_class=tokenizer,
    max_length=2048,
    apply_chat_template=True,   # Apply chat formatting
    packing=False,              # Keep conversations intact
    formatting_func=None,       # Custom identity formatting if needed
    dataset_num_proc=8,
    seed=42,
    cache_dir="./cache",
)

# Save processed instruction dataset
mixed_dataset.save_to_disk("./datasets/ft_dataset")
```

**Custom Identity Example:**
```python
def custom_formatting_func(example):
    """Add custom model identity"""
    conversation = [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I am an AI assistant named Doge, trained by the SmallDoge community based on the Doge architecture."},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {"conversation": conversation}

# Use with custom formatting
mixed_dataset = mix_datasets_by_radio(
    datasets_and_ratios=datasets_and_ratios,
    formatting_func=custom_formatting_func,
    # ...other parameters
)
```

**Benefits:**
- ✅ **Unified workflow**: Same pattern as pre-training
- ✅ **Chat template**: Automatic conversation formatting
- ✅ **Identity injection**: Easy to add custom model personality
- ✅ **Quality control**: Built-in filtering and validation

### 2. Supervised Fine-tuning (SFT)

Train the model to follow instructions:

```bash
# Run SFT training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/single_gpu.yaml \
    ./src/small_doge/trainer/doge/sft.py \
    --config recipes/doge/Doge-20M-Instruct/sft/config_full.yaml
```

**SFT Hyperparameters:**

| Model | Epochs | Context Length | Learning Rate | Batch Size |
|-------|--------|----------------|---------------|------------|
| Doge-20M | 2 | 2048 | 8e-4 | 0.25M |
| Doge-60M | 2 | 2048 | 6e-4 | 0.25M |
| Doge-160M | 2 | 2048 | 4e-4 | 0.25M |

### 3. Direct Preference Optimization (DPO)

Align model with human preferences:

```bash
# Run DPO training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/single_gpu.yaml \
    ./src/small_doge/trainer/doge/dpo.py \
    --config recipes/doge/Doge-20M-Instruct/dpo/config_full.yaml
```

**DPO Hyperparameters:**

| Model | Epochs | Context Length | Learning Rate | Batch Size |
|-------|--------|----------------|---------------|------------|
| Doge-20M | 2 | 1024 | 8e-5 | 0.125M |
| Doge-60M | 2 | 1024 | 6e-5 | 0.125M |
| Doge-160M | 2 | 1024 | 4e-5 | 0.125M |

## 🧠 Stage 3: Reasoning Fine-tuning (Reasoning Models)

Enhance models with reasoning capabilities using knowledge distillation and reinforcement learning.

### 1. Prepare Dataset (One-Stop Solution)

Use the processor for reasoning datasets:

```python
from transformers import AutoTokenizer
from small_doge.processor.ft_datasets_process import mix_datasets_by_radio

# Define reasoning datasets and ratios
datasets_and_ratios = [
    {"OpenThoughts-114k": 0.7},    # Reasoning traces for distillation
    {"OpenR1-Math-220k": 0.3},     # Mathematical reasoning for GRPO
]

# Load Doge tokenizer
tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")

# Process reasoning datasets with thinking prompts
mixed_dataset = mix_datasets_by_radio(
    datasets_and_ratios=datasets_and_ratios,
    total_sample_size=500000,   # 500K samples for reasoning training
    processing_class=tokenizer,
    max_length=4096,            # Longer context for reasoning traces
    apply_chat_template=True,
    apply_thinking_template=True,  # Enable thinking prompts
    packing=False,
    dataset_num_proc=8,
    seed=123,
    cache_dir="./cache",
)

# Save processed reasoning dataset
mixed_dataset.save_to_disk("./datasets/reasoning_dataset")
```

**Custom Thinking Prompt:**
```python
def reasoning_formatting_func(example):
    """Add thinking capabilities"""
    conversation = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": f"<thinking>\n{example['thinking']}\n</thinking>\n\n{example['response']}"},
    ]
    return {"conversation": conversation}

# Use with reasoning formatting
mixed_dataset = mix_datasets_by_radio(
    datasets_and_ratios=datasets_and_ratios,
    formatting_func=reasoning_formatting_func,
    # ...other parameters
)
```

> **Note**: For custom reasoning data generation, see [open-r1 project](https://github.com/huggingface/open-r1) to generate teacher model data using OpenAI's o1 or DeepSeek's R1.

### 2. Distillation Fine-tuning (DFT)

Learn reasoning patterns from teacher models:

```bash
# Run distillation training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/single_gpu.yaml \
    ./src/small_doge/trainer/doge/sft.py \
    --config recipes/doge/Doge-20M-Reason/sft/config_full.yaml
```

### 3. Group Relative Policy Optimization (GRPO)

Reinforce thinking abilities through RL:

```bash
# Run GRPO training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/single_gpu.yaml \
    ./src/small_doge/trainer/doge/grpo.py \
    --config recipes/doge/Doge-20M-Reason/grpo/config_full.yaml
```

## 🔧 Advanced Configuration

## 🔧 Advanced Configuration

### WSD Scheduler

SmallDoge uses a custom Warmup-Stable-Decay scheduler for seamless continued training:

```python
from small_doge.utils.scheduler import WSDScheduler

scheduler = WSDScheduler(
    optimizer=optimizer,
    warmup_steps=800,    # 10% of total steps
    stable_steps=6400,   # 80% of total steps (key innovation)
    decay_steps=800,     # 10% of total steps
    max_lr=8e-3,
    min_lr=8e-5
)
```

**Benefits:**
- **Stable Phase**: Allows continued training without learning rate spikes
- **Checkpoint Resume**: Can restart from any checkpoint in stable phase
- **Scaling**: Based on [compute-optimal training research](https://arxiv.org/pdf/2405.18392)

### Distributed Training

```bash
# Configure distributed setup
accelerate config

# Multi-GPU training
accelerate launch --num_processes 4 train.py

# DeepSpeed ZeRO-3 (Linux only)
accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    ./src/small_doge/trainer/doge/pt.py \
    --config recipes/doge/Doge-160M/config_full.yaml
```

### Memory Optimization

```python
training_args = {
    # Gradient checkpointing
    "gradient_checkpointing": True,
    
    # Mixed precision (choose one)
    "bf16": True,  # For Ampere GPUs (RTX 30/40 series)
    "fp16": True,  # For older GPUs
    
    # Batch size optimization
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    
    # Memory efficient settings
    "dataloader_pin_memory": False,
    "dataloader_num_workers": 0,
    
    # Advanced optimizations
    "torch_compile": True,  # PyTorch 2.0+ speedup
    "optim": "adamw_torch_fused",  # Fused optimizer
}
```

### Hardware Requirements

| Model Size | Min GPU Memory | Recommended GPU | Training Time | Inference Speed |
|------------|----------------|-----------------|---------------|-----------------|
| **20M** | 4GB | RTX 3060 | 14 hours | 142 tok/s (CPU) |
| **60M** | 8GB | RTX 3070 | 128 hours | 62 tok/s (CPU) |
| **160M** | 12GB | RTX 3080 | 522 hours | 28 tok/s (CPU) |
| **320M** | 16GB | RTX 4090 | 1856 hours | 16 tok/s (CPU) |

## 📊 Model Usage After Training

### Base Model Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-20M")
model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-20M", trust_remote_code=True)

inputs = tokenizer("Hey how are you doing?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.batch_decode(outputs))
```

### Instruction Model Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-20M-Instruct")
model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-20M-Instruct", trust_remote_code=True)

generation_config = GenerationConfig(
    max_new_tokens=100,
    use_cache=True,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.0
)

streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)
prompt = "Hi, how are you doing today?"

conversation = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    tokenize=True,
    return_tensors="pt",
)

outputs = model.generate(
    inputs,
    generation_config=generation_config,
    streamer=streamer
)
```

### Reasoning Model Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-20M-Reason")
model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-20M-Reason", trust_remote_code=True)

generation_config = GenerationConfig(
    max_new_tokens=1000,  # Longer for reasoning traces
    use_cache=True,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.0
)

streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)
prompt = "Solve this math problem step by step: What is 15% of 240?"

conversation = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    tokenize=True,
    return_tensors="pt",
)

outputs = model.generate(
    inputs,
    generation_config=generation_config,
    streamer=streamer
)
```

## 🔍 Model Evaluation

Use the lighteval toolkit to evaluate trained models:

```bash
# Install evaluation toolkit
pip install lighteval

# Linux evaluation
bash ./evaluation/eval_downstream_tasks.sh

# Windows evaluation  
powershell ./evaluation/eval_downstream_tasks.ps1
```

**Customize evaluation:**
```bash
# Set custom model and output directory
export MODEL="./path/to/your/model"
export OUTPUT_DIR="./eval_results"
bash ./evaluation/eval_downstream_tasks.sh
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size and enable gradient checkpointing
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --gradient_checkpointing True
   ```

2. **Training Instability**
   ```bash
   # Lower learning rate and increase warmup
   --learning_rate 4e-3 \
   --warmup_ratio 0.15 \
   --weight_decay 0.01
   ```

3. **Slow Training Speed**
   ```bash
   # Enable optimizations
   --bf16 True \
   --torch_compile True \
   --dataloader_num_workers 4
   ```

4. **DeepSpeed Issues (Windows)**
   ```bash
   # DeepSpeed not supported on Windows, use DDP instead
   accelerate launch --config_file recipes/accelerate_configs/ddp.yaml
   ```

5. **Dataset Download Errors**
   ```bash
   # Check internet connection and retry
   python ./examples/utils/download_pt_datasets.py --cache_dir ./cache --num_proc 1
   ```

### Debug Commands

```bash
# Enable debug logging
export ACCELERATE_LOG_LEVEL=debug
export TRANSFORMERS_VERBOSITY=debug

# Check GPU memory
nvidia-smi

# Monitor training progress
watch -n 1 nvidia-smi

# Check model loading
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('SmallDoge/Doge-20M', trust_remote_code=True)"
```

### Performance Tips

1. **Optimize Data Loading**
   ```python
   # Use multiple workers for data loading
   dataloader_num_workers = min(4, os.cpu_count())
   
   # Pin memory for GPU transfer
   dataloader_pin_memory = True if torch.cuda.is_available() else False
   ```

2. **Batch Size Tuning**
   ```python
   # Find optimal batch size
   def find_optimal_batch_size(model, tokenizer, device):
       for batch_size in [1, 2, 4, 8, 16]:
           try:
               # Test batch processing
               test_training_step(model, batch_size)
               print(f"Batch size {batch_size}: OK")
           except RuntimeError as e:
               if "out of memory" in str(e):
                   print(f"Batch size {batch_size}: OOM")
                   break
   ```

3. **Learning Rate Scheduling**
   ```python
   # Use learning rate finder
   from torch.optim.lr_scheduler import OneCycleLR
   
   scheduler = OneCycleLR(
       optimizer,
       max_lr=8e-3,
       total_steps=8000,
       pct_start=0.1,  # 10% warmup
       div_factor=25,  # initial_lr = max_lr/div_factor
       final_div_factor=1000  # final_lr = max_lr/final_div_factor
   )
   ```

## 📚 Additional Resources

- **[Model Architecture Details](./models.md)** - Deep dive into Doge architecture
- **[Dataset Preparation Guide](../recipes/doge/README.md)** - Detailed data processing
- **[Evaluation Methods](../evaluation/README.md)** - Benchmarking and testing
- **[WebUI Deployment](./webui.md)** - Setting up inference interface
- **[FAQ](./faq.md)** - Frequently asked questions

## 🤝 Community Support

- 💬 **Discord**: [Join our community](https://discord.gg/P2yYH95N)
- 🐛 **Issues**: [Report problems](https://github.com/SmallDoges/small-doge/issues)
- 💡 **Discussions**: [Share ideas](https://github.com/SmallDoges/small-doge/discussions)
- 📖 **Documentation**: [Contribute docs](https://github.com/SmallDoges/small-doge/tree/main/docs)
