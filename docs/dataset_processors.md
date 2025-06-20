# Dataset Processors

## 🎯 Overview

SmallDoge provides powerful dataset processing utilities for all training stages. These processors handle dataset mixing, format conversion, and preparation for different training tasks with consistent APIs and robust error handling.

## 📊 Available Processors

### 🔧 Pre-training Dataset Processor (`pt_datasets_process.py`)
Processes pre-training datasets with support for text tokenization, sequence packing, and efficient data loading.

### 💬 Supervised Fine-tuning Dataset Processor (`sft_datasets_process.py`)  
Processes supervised fine-tuning (SFT) datasets with conversation format conversion, ChatML template application, and tool support.

### 🎯 Direct Preference Optimization Dataset Processor (`dpo_datasets_process.py`)
Processes DPO (Direct Preference Optimization) training datasets, handling prompt, chosen, and rejected response pairs with tool integration.

## ✨ Key Features

All processors support the following functionality:

- **🔀 Dataset Mixing by Ratio**: Mix multiple datasets with specified ratios
- **✅ Dataset Validation**: Automatic validation of dataset format and ratio parameters
- **🔧 Tokenizer Validation**: Automatic configuration of tokenizer pad_token
- **🛡️ Error Handling**: Detailed error messages and exception handling
- **📊 Progress Logging**: Display processing progress and dataset statistics
- **💾 Caching Support**: Dataset caching for improved processing efficiency
- **🛠️ Tools Support**: Function calling support for SFT and DPO processors

## 🚀 Quick Start

### Python API Usage

```python
from small_doge.processor import mix_pt_datasets, mix_sft_datasets, mix_dpo_datasets
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")

# Define datasets and ratios
datasets_and_ratios = [
    {"SmallDoge/MiniCorpus:web-en": 0.6},
    {"SmallDoge/MiniCorpus:web-zh": 0.3},
    {"SmallDoge/MiniCorpus:code": 0.1},
]

# Mix pre-training datasets
mixed_dataset = mix_pt_datasets(
    datasets_and_ratios=datasets_and_ratios,
    total_sample_size=100000,
    dataset_text_field="text",
    processing_class=tokenizer,
    max_length=2048,
    packing=True,
    formatting_func=None,
    dataset_num_proc=4,
    seed=42,
    cache_dir="./cache",
)

# Save processed dataset
mixed_dataset.save_to_disk("./processed_dataset")
```

## 💻 Command Line Usage

### 🔧 Pre-training Dataset Processing
```bash
python pt_datasets_process.py \
    --datasets_and_ratios '[{"SmallDoge/MiniCorpus:web-en": 0.5}, {"SmallDoge/MiniCorpus:web-zh": 0.3}, {"SmallDoge/MiniCorpus:code": 0.2}]' \
    --dataset_save_path ./processed_pt_dataset \
    --total_sample_size 100000 \
    --dataset_text_field "text" \
    --tokenizer_name_or_path "SmallDoge/Doge-tokenizer" \
    --max_length 2048 \
    --packing \
    --dataset_num_proc 4 \
    --seed 42 \
    --cache_dir ./cache
```

### 💬 Supervised Fine-tuning Dataset Processing
```bash
python sft_datasets_process.py \
    --datasets_and_ratios '[{"SmallDoge/SmallTalks": 1.0}]' \
    --dataset_save_path ./processed_sft_dataset \
    --total_sample_size 50000 \
    --dataset_text_field "text" \
    --tokenizer_name_or_path "SmallDoge/Doge-tokenizer" \
    --max_length 2048 \
    --packing \
    --dataset_num_proc 4 \
    --seed 42 \
    --cache_dir ./cache \
    --tools '[{"type": "function", "function": {"name": "get_weather", "description": "Get weather info"}}]'
```

### 🎯 Direct Preference Optimization Dataset Processing
```bash
python dpo_datasets_process.py \
    --datasets_and_ratios '[{"SmallDoge/DPO-Pairs": 1.0}]' \
    --dataset_save_path ./processed_dpo_dataset \
    --total_sample_size 30000 \
    --tokenizer_name_or_path "SmallDoge/Doge-tokenizer" \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --dataset_num_proc 4 \
    --seed 42 \
    --cache_dir ./cache \
    --tools '[{"type": "function", "function": {"name": "get_weather", "description": "Get weather info"}}]'
```

## 📋 Parameter Reference

### ⚙️ Common Parameters

- `--datasets_and_ratios`: JSON string containing list of dictionaries with dataset names and mixing ratios
- `--dataset_save_path`: Path to save the processed dataset
- `--total_sample_size`: Total sample size for the mixed training dataset
- `--tokenizer_name_or_path`: Tokenizer name or path
- `--dataset_num_proc`: Number of processes for dataset processing
- `--seed`: Random seed for dataset shuffling
- `--cache_dir`: Directory for dataset caching

### 🔧 PT Specific Parameters

- `--dataset_text_field`: Name of the field containing text in the dataset
- `--max_length`: Maximum length of processed sequences
- `--packing`: Whether to use sequence packing for efficiency

### 💬 SFT Specific Parameters

- `--dataset_text_field`: Name of the field containing text in the dataset
- `--max_length`: Maximum length of processed sequences
- `--packing`: Whether to use sequence packing for efficiency
- `--tools`: Tools for chat template (JSON string)

### 🎯 DPO Specific Parameters

- `--max_prompt_length`: Maximum length of prompt sequences
- `--max_completion_length`: Maximum length of completion sequences
- `--tools`: Tools for chat template (JSON string)

## 📄 Dataset Format Requirements

### 🔧 PT Dataset
```json
{
    "text": "This is some pre-training text content..."
}
```

### 💬 SFT Dataset
Supports multiple formats:
```json
// Simple text format
{
    "text": "User: Hello\nAssistant: Hello, how can I help you?"
}

// Conversation format
{
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello, how can I help you?"}
    ]
}

// Prompt-completion format
{
    "prompt": "User: Hello\nAssistant:",
    "completion": "Hello, how can I help you?"
}
```

### 🎯 DPO Dataset
```json
{
    "prompt": "Explain what artificial intelligence is",
    "chosen": "Artificial intelligence is a field of computer science that aims to create machines that can simulate human intelligence...",
    "rejected": "AI is just robots..."
}
```

## ⚠️ Important Notes

1. **Dataset Ratios**: All dataset ratios must sum to 1.0
2. **Memory Usage**: When processing large datasets, monitor memory usage and reduce `dataset_num_proc` if needed
3. **Disk Space**: Processed datasets will require additional disk space
4. **Error Handling**: If a sample fails to process, the script will log the error and continue with empty tokenization
5. **Caching**: First run will download and cache datasets, subsequent runs will be faster

## 🔧 Troubleshooting

### ❌ Common Errors

1. **Ratios don't sum to 1.0**: Check that all ratios in `datasets_and_ratios` sum to 1.0
2. **Dataset not found**: Verify dataset names are correct and accessible
3. **Out of memory**: Reduce `dataset_num_proc` or `total_sample_size`
4. **Tokenizer issues**: Ensure tokenizer path is correct and compatible

### ⚡ Performance Optimization

1. Use SSD storage for cache directory to improve I/O performance
2. Adjust `dataset_num_proc` based on CPU core count
3. For large datasets, consider batch processing
4. Enabling `packing` improves training efficiency but increases processing time

## 📁 Project Structure

```
src/small_doge/processor/
├── pt_datasets_process.py      # Pre-training dataset processor
├── sft_datasets_process.py     # SFT dataset processor
├── dpo_datasets_process.py     # DPO dataset processor
└── __init__.py                 # Package initialization
```

## 📦 Dependencies

- `transformers`: For tokenizer support
- `datasets`: For dataset loading and processing
- `trl`: For TRL-specific data utilities
- `torch`: For tensor operations

## 🤝 Contributing

When adding new dataset processors:

1. Follow the same API pattern as existing processors
2. Include comprehensive error handling
3. Add validation for dataset format requirements
4. Update documentation with usage examples
5. Add tests for new functionality
