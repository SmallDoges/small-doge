---
library_name: transformers
license: apache-2.0
datasets:
- HuggingFaceTB/smoltalk
base_model:
- SmallDoge/Doge-20M
language:
- en
pipeline_tag: question-answering
tags:
- trl
- sft
- doge
---


# **Doge 20M Instruct SFT**

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

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-20M-Instruct-SFT")
model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-20M-Instruct-SFT", trust_remote_code=True)

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

We build the Doge-Instruct-SFT by SFT on [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).

**SFT**:
| Model | Training Data | Epochs | Content Length | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|
| [Doge-20M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-20M-Instruct-SFT) | [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 8e-4 | 0.25M | bfloat16 |
| [Doge-60M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-60M-Instruct-SFT) | [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 6e-4 | 0.25M | bfloat16 |
| [Doge-160M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-160M-Instruct-SFT) | [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 4e-4 | 0.25M | bfloat16 |
| [Doge-320M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-320M-Instruct-SFT) | [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 2e-4 | 0.25M | bfloat16 |


**Procedure**:

**SFT**:
[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/loser_cheems/huggingface/runs/eohr6fuj) 


**Environment**:
- Image: nvcr.io/nvidia/pytorch:24.12-py3
- Hardware: 1x NVIDIA RTX 4090
- Software: Transformers, TRL


## Citation

```bibtex
@misc{shi2024wonderfulmatrices,
      title={Wonderful Matrices: Combining for a More Efficient and Effective Foundation Model Architecture}, 
      author={Jingze Shi and Bingheng Wu},
      year={2024},
      eprint={2412.11834},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.11834}, 
}
```