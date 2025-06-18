---
library_name: transformers
license: apache-2.0
datasets:
- HuggingFaceTB/smoltalk
- HuggingFaceH4/ultrafeedback_binarized
base_model:
- SmallDoge/Doge-20M
language:
- en
pipeline_tag: question-answering
tags:
- trl
- sft
- dpo
- doge
---


# **Doge 20M Instruct**

<div align="center">
  <img src="https://huggingface.co/spaces/SmallDoge/README/resolve/main/org_icon.png" width="100%" alt="SmallDoge" />
</div>
<hr>
<div align="center">
  <a href="https://discord.gg/P2yYH95N" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-Small%20Doges-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <!-- <a href="https://arxiv.org/abs/2412.11834" target="_blank" style="margin: 2px;">
    <img alt="arXiv" src="https://img.shields.io/static/v1?label=arXiv&message=2412.11834&color=B31B1B&logo=arXiv" style="display: inline-block; vertical-align: middle;"/>
  </a> -->
  <a href="https://github.com/SmallDoges/small-doge" target="_blank" style="margin: 2px;">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-SmallDoge-181717?logo=github" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/SmallDoges/small-doge/blob/main/LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

Doge uses Dynamic Mask Attention as sequence transformation and can use Multi-Layer Perceptron or Cross Domain Mixture of Experts as state transformation. Dynamic Mask Attention allows the Transformer to use self-attention during training and state space during inference, and Cross Domain Mixture of Experts can directly inherit the weights of Multi-Layer Perceptron for further training. This model is trained by [SmallDoge](https://huggingface.co/SmallDoge) community, for detailed algorithm and model architecture, paper coming soon, all training details and code are available in the [small-doge](https://github.com/SmallDoges/small-doge) repository.


## Uses

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
steamer = TextStreamer(
      tokenizer=tokenizer, 
      skip_prompt=True
)

prompt = "Hi, how are you doing today?"
conversation = [
      {"role": "user", "content": prompt}
]
inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    tokenize=True,
    return_tensors="pt",
)

outputs = model.generate(
    inputs, 
    tokenizer=tokenizer,
    generation_config=generation_config, 
    streamer=steamer
)
```


## Model Details

We build the Doge-Instruct by first SFT on [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) and then DPO on [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).

**SFT**:
| Model | Training Data | Epochs | Content Length | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|
| [Doge-20M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-20M-Instruct-SFT) | [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 8e-4 | 0.25M | bfloat16 |
| [Doge-20M-MoE-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-20M-MoE-Instruct-SFT) | [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 8e-4 | 0.25M | bfloat16 |
| [Doge-60M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-60M-Instruct-SFT) | [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 6e-4 | 0.25M | bfloat16 |
| [Doge-120M-MoE-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-120M-MoE-Instruct-SFT) | [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 6e-4 | 0.25M | bfloat16 |
| [Doge-160M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-160M-Instruct-SFT) | [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 4e-4 | 0.25M | bfloat16 |
| [Doge-320M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-320M-Instruct-SFT) | [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 2e-4 | 0.25M | bfloat16 |

**DPO**:
| Model | Training Data | Epochs | Content Length | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|
| [Doge-20M-Instruct](https://huggingface.co/SmallDoge/Doge-20M-Instruct) | [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 8e-5 | 0.125M | bfloat16 |
| [Doge-20M-MoE-Instruct](https://huggingface.co/SmallDoge/Doge-20M-MoE-Instruct) | [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 8e-5 | 0.125M | bfloat16 |
| [Doge-60M-Instruct](https://huggingface.co/SmallDoge/Doge-60M-Instruct) | [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 6e-5 | 0.125M | bfloat16 |
| [Doge-120M-MoE-Instruct](https://huggingface.co/SmallDoge/Doge-120M-MoE-Instruct) | [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 6e-5 | 0.125M | bfloat16 |
| [Doge-160M-Instruct](https://huggingface.co/SmallDoge/Doge-160M-Instruct) | [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 4e-5 | 0.125M | bfloat16 |
| [Doge-320M-Instruct](https://huggingface.co/SmallDoge/Doge-320M-Instruct) | [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 2e-5 | 0.125M | bfloat16 |


**Evaluation**:

| Model | IFEval (Prompt Strict Acc) | MMLU | BBH | ARC | PIQA | HellaSwag | tokens / s on i7-11 CPU |
|---|---|---|---|---|---|---|---|
| [Doge-20M-Instruct](https://huggingface.co/SmallDoge/Doge-20M-Instruct) | 9.2 | 26.3 | 18.3 | 29.2 | 57.8 | 27.8 | 142 |
| [Doge-20M-MoE-Instruct](https://huggingface.co/SmallDoge/Doge-20M-MoE-Instruct) | 13.7 | 26.5 | 26.3 | 31.1 | 58.2 | 27.9 | 132 |
| [Doge-60M-Instruct](https://huggingface.co/SmallDoge/Doge-60M-Instruct) | 9.4 | 27.5 | 27.7 | 37.5 | 61.4 | 32.1 | 62 |
| [Doge-120M-MoE-Instruct](https://huggingface.co/SmallDoge/Doge-120M-MoE-Instruct) | 24.4 | 28.2 | 30.1 | 44.2 | 62.1 | 36.3 | 58 |
| [Doge-160M-Instruct](https://huggingface.co/SmallDoge/Doge-160M-Instruct) | 16.8 | 29.7 | 29.1 | 42.8 | 64.1 | 37.1 | 28 |
| [Doge-320M-Instruct](https://huggingface.co/SmallDoge/Doge-320M-Instruct) | 28.5 | 30.3 | 31.9 | 51.7 | 71.0 | 50.6 | 16 |


**Procedure**:

**SFT**:
[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/loser_cheems/huggingface/runs/eohr6fuj) 

**DPO**:
[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/loser_cheems/huggingface/runs/h6c2p2fe)


**Environment**:
- Image: nvcr.io/nvidia/pytorch:24.12-py3
- Hardware: 1x NVIDIA RTX 4090
- Software: Transformers, TRL


## Citation

```bibtex
@misc{smalldoges,
  title={SmallDoges: A Family of Dynamic UltraFast Small Language Models}, 
  author={Jingze, Shi and Yifan, Wu and Bingheng, Wu and Yuyu, Luo},
  year={2025},
  month={March},
  url={https://github.com/SmallDoges/small-doge}
}
```