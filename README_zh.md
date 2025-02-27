<div align="center">
   <img src="./assets/org_icon.png" alt="smalldoges" width="100%">
</div>

<hr>

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-Small%20Doges-7289da?logo=discord&logoColor=white&color=7289da)](https://discord.gg/P2yYH95N)
[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2412.11834&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2412.11834)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/SmallDoge/doge-slm-679cc991f027c4a3abbded4a)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

*Small Doges 正在建设中, 让我们一起开发吧!🐕🐕🐕*

<h4>

简体中文 | [English](./README.md)

</h4>

</div>

# small-doge

**新闻**: 🎉🎉🎉 我们现在已经支持 **预训练的Doge-Base**, **指令微调的Doge-Instruct**, 以及**推理微调的Doge-R1**的全部训练流程, 请参阅[指南](./recipes/doge/README_zh.md)!

* 本项目旨在从**0**开始, 最快仅用3小时！即可训练出仅为13M大小的微型语言模型[Doge-20M](https://huggingface.co/SmallDoge/Doge-20M)!🚀
* small doge系列极其轻量, 最小版本体积约是 GPT3 的 **$\frac{1}{7800}$**, 力求做到最普通的个人GPU也可快速推理甚至训练.🏎️
* 我们提供了数据集预处理、预训练、监督微调、强化学习偏好对齐的全阶段代码、视觉多模态VLM(正在开发)和推理微调R1(正在开发).🧪
* 站在巨人的肩膀上可以看的更远, 希望small doge系列小模型能为研究者提供更多思路, 为实现**具身通用人工智能**的道路添砖加瓦.🤖

> [!TIP]
> 我们希望尽可能使用开源工具和框架来简化从数据处理到模型训练的过程, 以便初学者可以轻松理解和使用.🤗

<img src="./assets/reasoning.gif" alt="streamlit"/>
<figcaption>Doge-60M-Instruct 在 11 代 i7 CPU 笔记本上快速推理</figcaption>


## 关于

本项目旨在开发一系列动态快速的小型模型, 以促进其在具身智能领域的应用, 特别是在资源受限的环境下, 满足实时响应需求, 推动下游领域的实际应用落地.

> [!TIP]
> *截至2025-2-20*: small doge系列已完成了3个型号模型的预训练, 最小仅需20M, 即可具备流畅的对话能力!

| 模型 | 代币 | 训练步数 | 批次 | 学习率 | 调度器 | 预热比例 | 衰减比例 | 权重衰减 | 最小学习率 |
|---|---|---|---|---|---|---|---|---|---|
| Doge-20M | 4B | 8,000 | 256 | 8e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |
| Doge-60M | 16B | 16,000 | 512 | 6e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |
| Doge-160M | 32B | 24,000 | 768 | 4e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |

> 以下一个型号正在预训练, 欢迎有能力的研究员帮忙(poor man的哀嚎)!🙏

| 模型 | 代币 | 训练步数 | 批次 | 学习率 | 调度器 | 预热比例 | 衰减比例 | 权重衰减 | 最小学习率 |
|---|---|---|---|---|---|---|---|---|---|
| Doge-320M | 64B | 32,000 | 1024 | 2e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |

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


## 安装要求

如果您需要预训练或者微调, 我们的代码库需要以下环境:

- Windows 或 Linux
- NVIDIA GPU
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

但我们仍然强烈建议您安装最新版本的 PyTorch 和 CUDA 以获得最佳性能.

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
git clone https://github.com/SmallDoges/small-doge.git
cd small-doge
pip install -e .
```


## 快速入门

我们已经编写了一个 [notebook](./examples/notebook.ipynb) 和 [训练指南](./recipes/doge/README.md) 来演示数据集处理、模型训练和模型评估的整个过程. 您还可以独立使用已经发布的模型, 如果感兴趣请详细阅读notebook或训练指南, 里面有具体的步骤和细节！


## 型号发布

### Doge-CheckPoint

![wsd_scheduler](./assets/wsd_scheduler.png)

Doge 使用 `wsd_scheduler` 作为训练调度器, 将学习率分为 `warmup`, `stable` 和 `decay` 三个阶段. 它允许我们在 `stable stage` 中的任何新数据集上从任何检查点继续训练, 而没有训练的损失波动.

以下是在每个检查点继续训练所需的初始学习率:

- **[Doge-20M](https://huggingface.co/SmallDoge/Doge-20M-checkpoint)**: 8e-3
- **[Doge-60M](https://huggingface.co/SmallDoge/Doge-60M-checkpoint)**: 6e-3
- **[Doge-160M](https://huggingface.co/SmallDoge/Doge-160M-checkpoint)**: 4e-3
- **Doge-320M**: 2e-3

| 模型 | 学习率 | 调度器 | 预热步数 | 稳定步数 |
|---|---|---|---|---|
| [Doge-20M]((https://huggingface.co/SmallDoge/Doge-20M-checkpoint)) | 8e-3 | wsd_scheduler | 800 | 6400 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M-checkpoint) | 6e-3 | wsd_scheduler | 1600 | 12800 |
| [Doge-160M](https://huggingface.co/SmallDoge/Doge-160M-checkpoint) | 4e-3 | wsd_scheduler | 2400 | 19200 |
| Doge-320M | 2e-3 | wsd_scheduler | 3200 | 25600 |

### Doge-Base

**预训练**:
| 模型 | 训练数据 | 步数 | 上下文长度 | 令牌 | 学习率 | 批量大小 | 精度 | RTX 4090 GPU 小时 |
|---|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 8k  | 2048 | 4B | 8e-3 | 0.5M | bfloat16 | 14 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 16k  | 2048 | 16B | 6e-3 | 1M | bfloat16 | 128 |
| [Doge-160M](https://huggingface.co/SmallDoge/Doge-160M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 24k  | 2048 | 32B | 4e-3 | 1.5M | bfloat16 | 522 |

**评估**:
| 模型 | MMLU | TriviaQA | ARC | PIQA | HellaSwag | OBQA | Winogrande | i7-11 CPU上的tokens/s |
|---|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M) | 25.4 | 0.03 | 29.8 | 58.4 | 27.3 | 25.6 | 50.2 | 142 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M) | 26.4 | 0.2 | 37.9 | 61.4 | 31.5 | 28.0 | 50.8 | 62 |
| [Doge-160M](https://huggingface.co/SmallDoge/Doge-160M) | 29.2 | 4.8 | 44.4 | 66.3 | 38.7 | 34.4 | 52.2 | 28 |


### Doge-Instruct

**监督微调**:
| 模型 | 训练数据 | 轮次 | 上下文长度 | 学习率 | 批量大小 | 精度 |
|---|---|---|---|---|---|---|
| [Doge-20M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-20M-Instruct-SFT) | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 8e-4 | 0.25M | bfloat16 |
| [Doge-60M-Instruct-SFT](https://huggingface.co/SmallDoge/Doge-60M-Instruct-SFT) | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 2048 | 6e-4 | 0.25M | bfloat16 |

**直接优化微调**:
| 模型 | 训练数据 | 轮次 | 上下文长度 | 学习率 | 批量大小 | 精度 |
|---|---|---|---|---|---|---|
| [Doge-20M-Instruct](https://huggingface.co/SmallDoge/Doge-20M-Instruct) | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 8e-5 | 0.125M | bfloat16 |
| [Doge-60M-Instruct](https://huggingface.co/SmallDoge/Doge-60M-Instruct) | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 2 | 1024 | 6e-5 | 0.125M | bfloat16 |

**评估**:
| 模型 | IFEval (Prompt Strict Acc) | MMLU | BBH | ARC | PIQA | HellaSwag | tokens / s on i7-11 CPU |
|---|---|---|---|---|---|---|---|
| [Doge-20M-Instruct](https://huggingface.co/SmallDoge/Doge-20M-Instruct) | 7.3 | 26.3 | 18.3 | 29.2 | 57.8 | 27.8 | 142 |
| [Doge-60M-Instruct](https://huggingface.co/SmallDoge/Doge-60M-Instruct) | 7.4 | 27.5 | 27.7 | 37.5 | 61.4 | 32.1 | 62 |
| [Doge-160M-Instruct](https://huggingface.co/SmallDoge/Doge-160M-Instruct) | 16.8 | 29.7 | 29.1 | 42.8 | 64.1 | 37.1 | 28 |


**训练环境**:

- 镜像: nvcr.io/nvidia/pytorch:24.12-py3
- 硬件: 1x NVIDIA RTX 4090
- 软件: Transformers, TRL


## 期许
> [!IMPORTANT]
> - 如果您觉得 small-doge 对你有所帮助, 请给它一颗星星⭐吧! <br><br>
> - 由于时间和专业知识的限制, 项目中可能会有疏漏.
> 欢迎在Issue提出你的见解或通过提交PR来帮助项目不断完善, 您的支持是推动项目持续进步的动力源泉!😊 <br><br>
> - 一个人可以走的很快, 一群人才可以走的更远. 如果您已经尝试训练了新的 small-doge 型号，欢迎在 Discussions 或 Issues 中分享您的模型权重、训练配方、评估结果和其他相关信息. 可以是在特定下游任务或垂直领域, 例如情感识别、医疗、心理、金融、法律问答等的small-doge新模型版本. 也可以是拓展训练, 例如探索更长文本序列、更大参数或更大的数据集的small-doge新模型版本. 您的分享将会大力促进社区的发展!🚀🚀🚀


## 星星历史

[![Star History Chart](https://api.star-history.com/svg?repos=SmallDoges/small-doge&type=Date)](https://star-history.com/#SmallDoges/small-doge&Date)


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