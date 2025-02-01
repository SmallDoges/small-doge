<div align="center">
    <img src="./assets/org_icon.png" alt="samlldoges" width="100%">
</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=SmallDoges/small-doge)
[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2412.11834&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2412.11834)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/SmallDoge/doge-slm-679cc991f027c4a3abbded4a)
[![GitHub last commit](https://img.shields.io/github/last-commit/SmallDoges/small-doge)](https://github.com/SmallDoges/small-doge/commits/master)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/SmallDoges/small-doge/pulls)


<div align="center">
  <h3>"Small Doges is under construction, let's develop together!"</h3>
</div>

<div align="center">

English | [简体中文](./README_zh.md)

</div>

* This open-source project aims to start completely from scratch, and in just 3 hours, you can train a micro language model **Doge** with a size of only 13M.
* The **small doge** series is extremely lightweight, with the smallest version being approximately $\frac{1}{7877}$ the size of GPT-3, striving to enable fast inference and even training on the most ordinary personal GPUs.
* **small doge** has released the full-stage code for the large model doge structure, dataset cleaning and preprocessing, supervised pre-training (Pretrain), supervised fine-tuning (SFT), direct preference optimization (DPO) without reward reinforcement learning, and visual multimodal VLM (under development).
* Standing on the shoulders of giants allows us to see further. We hope that the small doge series of small models can provide more ideas for researchers and contribute to the path of achieving AGI.

  > We hope to simplify the process from data processing to model training as much as possible using open-source tools and frameworks, so that beginners can easily understand and use them. Detailed specifications will be provided below.

<div align="center">
    <img src="./assets/reasoning.gif" alt="streamlit"/>
    <figcaption>Doge-60M-Instruct CPU Fast Inference</figcaption>
</div>

---

# 📌 Introduction

Currently, large language models such as GPT-4 and Deepseek have achieved remarkable results in the field of natural language processing and have attracted significant attention. However, in practical applications and research, large models face many issues. On one hand, they require high computational infrastructure, making it difficult to deploy in resource-constrained scenarios such as mobile devices and edge computing, and they have poor interpretability, limiting their application in fields like healthcare and finance. On the other hand, large-scale investments raise the research threshold, and training data often contain biases and prejudices.

Meanwhile, embodied intelligence has broad development prospects, emphasizing intelligent agents interacting with the physical environment to achieve intelligent decision-making, with great potential in fields such as smart homes and logistics robots. However, due to their inherent limitations, large language models struggle with tasks related to embodied intelligence.

In contrast, small models have unique advantages in the development of embodied intelligence. They require fewer computational resources and can run on ordinary or small embedded devices, allowing intelligent agents to work flexibly in resource-constrained environments and meet real-time response needs. Moreover, small models have simple structures and strong interpretability, making it easier for engineers to link model decisions with agent actions and optimize their behavior. Additionally, small models require less training data, reducing data bias and facilitating precise training of intelligent agents in specific scenarios.

Therefore, the goal of this project is to create a series of efficient, fast, and stable small models to promote the application and implementation in downstream fields as much as possible.

> [!TIP]
>  (As of 2025-2-2) The small doge series has completed the pre-training of two model types, with the smallest requiring only 20M (0.02B) and capable of fluent conversation!

| Model | tokens | max_train_steps | accumulate_steps | learning_rate | scheduler | warmup_ratio | decay_ratio | weight_decay | min_lr_rate |
|---|---|---|---|---|---|---|---|---|---|
| Doge-20M | 4B | 8,000 | 256 | 8e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |
| Doge-60M | 16B | 16,000 | 512 | 6e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 | |

> The above two models are currently in pre-training (researchers with the capability are welcome to help)

| Model | tokens | max_train_steps | accumulate_steps | learning_rate | scheduler | warmup_ratio | decay_ratio | weight_decay | min_lr_rate |
|---|---|---|---|---|---|---|---|---|---|
| Doge-160M | 32B | 24,000 | 768 | 4e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |
| Doge-320M | 64B | 32,000 | 1024 | 2e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |


<div align="center">
    <img src="./assets/doge_architecture.png" alt="drawing" width="600"/>
</div>

As shown in the figure, the sequence transformation part of the Doge architecture uses `Dynamic Mask Attention`, which can be understood as using self-attention related to value states during training, and using state-space without past state decay during inference, to solve the problem of existing Transformers or SSMs getting lost in long text. The state transformation part of Doge uses `Cross Domain Mixture of Experts`, which consists of dense linear layers and sparse embedding layers, and can additionally increase sparse parameters to continue training from dense weight checkpoints without retraining the entire model, thereby reducing the cost of continuous iteration of the model. In addition, Doge also uses `RMSNorm` and `Residual` with learnable parameters to adapt the gradient range of deep models.

**Dynamic Mask Attention Module**

![DMAttn](./assets/dmattn.png)
![DMAttn](./assets/mqar.png)

**Cross Domain Mixture of Experts Module**

![CDMoE](./assets/cdmoe.png)
![CDMoE](./assets/merm.png)

We also hope to use open-source tools and frameworks as much as possible to simplify the process from data processing to model training, so that beginners can easily understand and use them.


# 📌 environment
This is just a personal hardware and software configuration, please adjust as needed:

- Windows or Linux
- NVIDIA GPU
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

We highly recommend that you install the latest version of PyTorch and CUDA for optimal performance.

Of course, you can also use the open-source [Docker PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) image to avoid the hassle of configuring the environment.

```bash
docker pull nvcr.io/nvidia/pytorch:24.12-py3
docker run --privileged --gpus all -it --name PyTorch --shm-size=32g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 -v <your code path>:/workspace -v <your datasets path>:/workspace/Doge/datasets nvcr.io/nvidia/pytorch:24.12-py3
```

- `pip install transformers`: The core framework for all subsequent work.
- `pip install datasets sentencepiece boto3`: Used to download and process datasets.
- `pip install accelerate`: Used for distributed training.
- `pip install trl`: Used for fine-tuning with reinforcement learning.


## Installation

```bash
git clone https://github.com/SamllDoge/small-doge.git
cd small-doge
pip install -e .
```


## Quick Start Train

We have written a [notebook](./examples/notebook.ipynb) (still being updated) to demonstrate the entire process of datasets processing, model training, and model evaluation. You can also use the models that have been released independently,If you are interested, please read the notebook in detail, which contains specific steps and details!


# 📌Models Released

### Doge-CheckPoint

![wsd_scheduler](./assets/wsd_scheduler.png)

Doge uses `wsd_scheduler` as the training scheduler, which divides the learning rate into three stages: `warmup`, `stable`, and `decay`. It allows us to continue training on any new dataset from any checkpoint in the `stable stage` without spikes of the training.

Here are the initial learning rates required to continue training at each checkpoint:

- **[Doge-20M](https://huggingface.co/SmallDoge/Doge-20M-checkpoint)**: 8e-3
- **[Doge-60M](https://huggingface.co/SmallDoge/Doge-60M-checkpoint)**: 6e-3
- **[Doge-160M](https://huggingface.co/SmallDoge/Doge-160M-checkpoint)**: 4e-3
- **Doge-320M**: 2e-3

| Model | Learning Rate | Schedule | Warmup Steps | Stable Steps |
|-------|---------------|----------|--------------|--------------|
| [Doge-20M]((https://huggingface.co/SmallDoge/Doge-20M-checkpoint)) | 8e-3 | wsd_scheduler | 800 | 6400 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M-checkpoint) | 6e-3 | wsd_scheduler | 1600 | 12800 |
| [Doge-160M](https://huggingface.co/SmallDoge/Doge-160M-checkpoint) | 4e-3 | wsd_scheduler | 2400 | 19200 |
| Doge-320M | 2e-3 | wsd_scheduler | 3200 | 25600 ||

### Doge-SLM

**Pre-Training**:
| Model | Training Data | Steps | Content Length | Tokens | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 8k  | 2048 | 4B | 8e-3 | 0.5M | bfloat16 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 16k  | 2048 | 16B | 6e-3 | 1M | bfloat16 |

**Evaluation**:
| Model | MMLU | TriviaQA | ARC-E | ARC-C | PIQA | HellaSwag | OBQA | Winogrande | tokens / s on CPU |
|---|---|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M) | 25.43 | 0.03 | 36.83 | 22.78 | 58.38 | 27.25 | 25.60 | 50.20 | 142 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M) | 26.41 | 0.18 | 50.46 | 25.34 | 61.43 | 31.45 | 28.00 | 50.75 | 62 |

> All evaluations are done using five-shot settings, without additional training on the benchmarks.

**SFT**:
| Model | Training Data | Epochs | Content Length | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|
| [Doge-20M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-20M-Instruct-SFT) | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 8e-4 | 0.25M | bfloat16 |
| [Doge-60M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-60M-Instruct-SFT) | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 6e-4 | 0.25M | bfloat16 |

**DPO**:
| Model | Training Data | Epochs | Content Length | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|
| [Doge-20M-Instruct](https://huggingface.co/SmallDoge/Doge-20M-Instruct) | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 8e-5 | 0.125M | bfloat16 |
| [Doge-60M-Instruct](https://huggingface.co/SmallDoge/Doge-60M-Instruct) | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 6e-5 | 0.125M | bfloat16 |

**Environment**:
- Image: nvcr.io/nvidia/pytorch:24.12-py3
- Hardware: 1x NVIDIA RTX 4090
- Software: Transformers, TRL


# 📌引用

If you use this codebase, or otherwise find our work valuable, please cite our paper:

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
