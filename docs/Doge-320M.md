---
library_name: transformers
license: apache-2.0
datasets:
- HuggingFaceTB/smollm-corpus
language:
- en
pipeline_tag: text-generation
tags:
- pt
- doge
---


# **Doge 320M**

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
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-320M")
>>> model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-320M", trust_remote_code=True)
>>> inputs = tokenizer("Hey how are you doing?", return_tensors="pt")

>>> out = model.generate(**inputs, max_new_tokens=100)
>>> print(tokenizer.batch_decode(out))
```


## Model Details

We build the Doge by doing Per-Training on [Smollm-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus). If you want to continue pre-training this model, you can find the unconverged checkpoint [here](https://huggingface.co/SmallDoge/Doge-320M-checkpoint). These models has not been fine-tuned for instruction, the instruction model is [here](https://huggingface.co/SmallDoge/Doge-320M-Instruct).


**Pre-Training**:

| Model | Training Data | Steps | Content Length | Tokens | LR | Batch Size | Precision | RTX 4090 GPU hours |
|---|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M) | [smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 8k  | 2048 | 4B | 8e-3 | 0.5M | bfloat16 | 14 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M) | [smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 16k  | 2048 | 16B | 6e-3 | 1M | bfloat16 | 128 |
| [Doge-160M](https://huggingface.co/SmallDoge/Doge-160M) | [smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 24k  | 2048 | 32B | 4e-3 | 1.5M | bfloat16 | 522 |
| [Doge-320M](https://huggingface.co/SmallDoge/Doge-320M) | [smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 32k  | 2048 | 64B | 2e-3 | 2M | bfloat16 | 1856 |

**Evaluation**:

| Model | MMLU | TriviaQA | ARC | PIQA | HellaSwag | OBQA | Winogrande | tokens / s on i7-11 CPU |
|---|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M) | 25.4 | 0.03 | 29.8 | 58.4 | 27.3 | 25.6 | 50.2 | 142 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M) | 26.4 | 0.2 | 37.9 | 61.4 | 31.5 | 28.0 | 50.8 | 62 |
| [Doge-160M](https://huggingface.co/SmallDoge/Doge-160M) | 29.2 | 4.8 | 44.4 | 70.1 | 43.4 | 34.4 | 52.2 | 28 |
| [Doge-320M](https://huggingface.co/SmallDoge/Doge-320M) | 35.6 | 9.4 | 55.4 | 73.9 | 52.7 | 37.9 | 59.3 | 16 |

**Procedure**:

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/loser_cheems/huggingface/runs/y18ty3sh) 

**Environment**:

- Image: nvcr.io/nvidia/pytorch:24.12-py3
- Hardware: 1x NVIDIA RTX 4090
- Software: Transformers


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