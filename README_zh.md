<div align="center">
    <img src="./assets/org_icon.png" alt="samlldoges" width="100%">
</div>

<div align="center">

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2412.11834&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2412.11834)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/SmallDoge/doge-slm-679cc991f027c4a3abbded4a)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

<h4>

**Small Doges 正在建设中, 让我们一起开发吧!**

[English](./README.md) | 简体中文

</h4>

</div>


## 关于

<div align="center">
    <img src="./assets/doge_architecture.png" alt="drawing" width="600"/>
</div>

如图所示, Doge 架构的序列变换部分使用了 `Dynamic Mask Attention`, 可以理解为在训练时使用与值状态相关的自注意力, 在推理时使用没有过去状态衰减的状态空间, 以解决现有的 Transformer 或 SSM 在长文本中迷失的问题. Doge 的状态变换部分使用了 `Cross Domain Mixture of Experts`, 由密集线性层和稀疏嵌入层组成, 并可以额外增加稀疏参数, 以从密集权重检查点继续训练而无需重新训练整个模型, 从而降低模型的持续迭代成本. 此外, Doge 还使用了具有可学习参数的 `RMSNorm` 和 `Residual` 来适应深度模型的梯度范围.

**Dynamic Mask Attention 模块**

![DMAttn](./assets/dmattn.png)
![DMAttn](./assets/mqar.png)

**Cross Domain Mixture of Experts 模块**

![CDMoE](./assets/cdmoe.png)
![CDMoE](./assets/merm.png)

我们还希望尽可能使用开源工具和框架来简化从数据处理到模型训练的过程, 以便初学者可以轻松理解和使用.


## 要求

- Windows 或 Linux
- NVIDIA GPU
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

我们强烈建议您安装最新版本的 PyTorch 和 CUDA 以获得最佳性能.

当然, 您也可以使用开源的 [Docker PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) 镜像来避免配置环境的麻烦.

```bash
docker pull nvcr.io/nvidia/pytorch:24.12-py3
docker run --privileged --gpus all -it --name PyTorch --shm-size=32g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 -v <your code path>:/workspace -v <your datasets path>:/workspace/Doge/datasets nvcr.io/nvidia/pytorch:24.12-py3
```

- `pip install transformers`: 所有后续工作的核心框架.
- `pip install datasets sentencepiece boto3`: 用于下载和处理数据集.
- `pip install accelerate`: 用于分布式训练.
- `pip install trl`: 用于强化学习微调.


## 安装

```bash
git clone https://github.com/SamllDoge/SmallDoges.git
cd SmallDoges
pip install -e .
```


## 使用

我们已经编写了一个 [notebook](./examples/notebook.ipynb) (仍在更新) 来演示数据集处理、模型训练和模型评估的整个过程. 您还可以独立使用已经发布的模型.


## 已发布的模型

### Doge-CheckPoint

![wsd_scheduler](./assets/wsd_scheduler.png)

Doge 使用 `wsd_scheduler` 作为训练调度器, 将学习率分为 `warmup`, `stable` 和 `decay` 三个阶段. 它允许我们在 `stable stage` 中的任何新数据集上从任何检查点继续训练, 而没有训练的损失波动.

以下是在每个检查点继续训练所需的初始学习率:

- **[Doge-20M](https://huggingface.co/SmallDoge/Doge-20M-checkpoint)**: 8e-3
- **[Doge-60M](https://huggingface.co/SmallDoge/Doge-60M-checkpoint)**: 6e-3
- **[Doge-160M](https://huggingface.co/SmallDoge/Doge-160M-checkpoint)**: 4e-3
- **Doge-320M**: 2e-3

| 模型 | 学习率 | 调度器 | 预热步数 | 稳定步数 |
|-------|---------------|----------|--------------|--------------|
| [Doge-20M]((https://huggingface.co/SmallDoge/Doge-20M-checkpoint)) | 8e-3 | wsd_scheduler | 800 | 6400 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M-checkpoint) | 6e-3 | wsd_scheduler | 1600 | 12800 |
| [Doge-160M](https://huggingface.co/SmallDoge/Doge-160M-checkpoint) | 4e-3 | wsd_scheduler | 2400 | 19200 |
| Doge-320M | 2e-3 | wsd_scheduler | 3200 | 25600 |

### Doge-SLM

**预训练**:
| 模型 | 训练数据 | 步数 | 上下文长度 | 令牌 | 学习率 | 批量大小 | 精度 |
|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 8k  | 2048 | 4B | 8e-3 | 0.5M | bfloat16 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 16k  | 2048 | 16B | 6e-3 | 1M | bfloat16 |

**评估**:
| 模型 | MMLU | TriviaQA | ARC-E | ARC-C | PIQA | HellaSwag | OBQA | Winogrande | CPU 上的 tokens / s |
|---|---|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M) | 25.43 | 0.03 | 36.83 | 22.78 | 58.38 | 27.25 | 25.60 | 50.20 | 142 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M) | 26.41 | 0.18 | 50.46 | 25.34 | 61.43 | 31.45 | 28.00 | 50.75 | 62 |

> 所有评估都是在five-shot设置下完成的, 在基准测试中没有额外的训练.

**监督微调**:
| 模型 | 训练数据 | 轮次 | 上下文长度 | 学习率 | 批量大小 | 精度 |
|---|---|---|---|---|---|---|
| [Doge-20M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-20M-Instruct-SFT) | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 8e-4 | 0.25M | bfloat16 |
| [Doge-60M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-60M-Instruct-SFT) | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 6e-4 | 0.25M | bfloat16 |

**近端优化微调**:
| 模型 | 训练数据 | 轮次 | 上下文长度 | 学习率 | 批量大小 | 精度 |
|---|---|---|---|---|---|---|
| [Doge-20M-Instruct](https://huggingface.co/SmallDoge/Doge-20M-Instruct) | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 8e-5 | 0.125M | bfloat16 |
| [Doge-60M-Instruct](https://huggingface.co/SmallDoge/Doge-60M-Instruct) | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 6e-5 | 0.125M | bfloat16 |

**环境**:
- 镜像: nvcr.io/nvidia/pytorch:24.12-py3
- 硬件: 1x NVIDIA RTX 4090
- 软件: Transformers, TRL


## 引用

如果您使用此代码库, 或者认为我们的工作有价值, 请引用我们的论文:

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