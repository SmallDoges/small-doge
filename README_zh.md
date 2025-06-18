<div align="center">
   <img src="./assets/org_icon.png" alt="SmallDoges" width="100%">
</div>

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-Small%20Doges-7289da?logo=discord&logoColor=white&color=7289da)](https://discord.gg/P2yYH95N)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/SmallDoge/doge-slm-679cc991f027c4a3abbded4a)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**简体中文** | [English](./README.md)

</div>

# SmallDoge: 超快小型语言模型

> **仅需3小时即可训练20M参数语言模型！** 🚀

SmallDoge是一系列动态、超快的小型语言模型，专注于**效率**和**易用性**。

## ✨ 核心特性

- **🚀 超快训练**: 20M模型仅需3小时训练
- **💡 创新架构**: 动态掩码注意力 + 跨域专家混合
- **🏎️ 闪电推理**: i7-11 CPU上142 tokens/秒
- **🔧 完整工具链**: 预训练 → 指令微调 → 推理微调
- **🌐 Web界面**: 内置聊天界面和OpenAI兼容API

<div align="center">
    <img src="./assets/reasoning.gif" alt="Doge-60M-Instruct演示" width="60%"/>
    <br><em>Doge-60M-Instruct在i7-11 CPU上运行</em>
</div>


## 🚀 快速开始

### 安装

```bash
git clone https://github.com/SmallDoges/small-doge.git
cd small-doge
pip install -e .
```

### 基础使用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
model_name = "SmallDoge/Doge-60M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 生成文本
prompt = "请用简单的语言解释机器学习："
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Web界面

```bash
# 安装WebUI
pip install -e '.[webui]'

# 启动界面
small-doge-webui
```

**访问地址**: http://localhost:7860 (前端) | http://localhost:8000 (API)

📖 **详细指南**: [快速开始](./docs/quickstart.md) | [安装说明](./docs/installation.md) | [训练指南](./docs/training.md)

## 📊 可用模型

| 模型 | 参数量 | 速度 (i7-11 CPU) | MMLU | 适用场景 |
|------|-------|------------------|------|----------|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M) | 20M | 142 tok/s | 25.4 | 超快原型开发 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M) | 60M | 62 tok/s | 26.4 | 平衡性能 |
| [Doge-160M](https://huggingface.co/SmallDoge/Doge-160M) | 160M | 28 tok/s | 29.2 | 更好推理能力 |
| [Doge-320M](https://huggingface.co/SmallDoge/Doge-320M) | 320M | 16 tok/s | 33.8 | 生产就绪 |

**指令模型**: 在任何模型名后添加`-Instruct`即可获得聊天优化版本。

**检查点**: 添加`-checkpoint`用于继续训练（参见[模型文档](./docs/models.md)）。

## 🏗️ 架构创新

<div align="center">
    <img src="./assets/doge_architecture.png" alt="Doge架构" width="70%"/>
</div>

**核心创新：**
- **动态掩码注意力**: 高效处理长序列的混合注意力机制
- **跨域专家混合**: 支持密集到稀疏的继续训练
- **WSD调度器**: 热身-稳定-衰减，实现无缝检查点恢复


## 🎓 训练流程

SmallDoge支持完整的三阶段训练：

1. **预训练** → 基础模型 (Doge-Base)
2. **指令微调** → 聊天模型 (Doge-Instruct) 
3. **推理微调** → 推理模型 (Doge-Reason)

**核心特性：**
- 🚀 **一站式处理器**: 三阶段统一数据处理
- 🔧 **灵活配方**: 预配置训练配置文件
- 📊 **高效训练**: 针对小模型优化
- 🔄 **无缝继续**: WSD调度器支持检查点恢复

**训练时间** (RTX 4090):
- Doge-20M: 14小时 | Doge-60M: 128小时 | Doge-160M: 522小时 | Doge-320M: 1856小时

📚 **了解更多**: [训练指南](./docs/training.md)

## 📈 评估结果

### 基础模型
| 模型 | MMLU | ARC | PIQA | HellaSwag | Winogrande |
|------|------|-----|------|-----------|------------|
| Doge-20M | 25.4 | 29.8 | 58.4 | 27.3 | 50.2 |
| Doge-60M | 26.4 | 37.9 | 61.4 | 31.5 | 50.8 |
| Doge-160M | 29.2 | 44.4 | 70.1 | 43.4 | 52.2 |
| Doge-320M | 33.8 | 52.1 | 73.9 | 52.7 | 55.0 |

### 指令模型
| 模型 | IFEval | MMLU | BBH | 性能表现 |
|------|--------|------|-----|----------|
| Doge-20M-Instruct | 7.3 | 26.3 | 18.3 | 适合基础聊天 |
| Doge-60M-Instruct | 7.4 | 27.5 | 27.7 | 平衡聊天模型 |
| Doge-160M-Instruct | 16.8 | 29.7 | 29.1 | 高级推理能力 |

🔍 **评估工具包**: [评估指南](./docs/evaluation.md)

## 🛠️ 应用场景

- **🤖 边缘AI**: 部署在资源受限设备上
- **🎮 游戏**: 实时NPC对话和游戏机制
- **📱 移动应用**: 设备端AI助手
- **🔬 科研**: 快速原型开发和实验
- **📚 教育**: 使用可管理的模型学习AI/ML
- **🏭 工业**: 轻量级生产部署

## 📦 项目结构

```
small-doge/
├── src/small_doge/          # 核心实现
│   ├── models/              # 模型架构
│   ├── trainer/             # 训练代码
│   ├── processor/           # 数据处理
│   └── webui/               # Web界面
├── recipes/                 # 训练配方
│   └── doge/                # Doge模型配置
├── examples/                # 教程和示例
├── evaluation/              # 评估工具包
├── docs/                    # 文档
└── assets/                  # 图片和资源
```


## 🤝 贡献

我们欢迎贡献！您可以这样帮助我们：

- 🐛 **报告错误**: [GitHub Issues](https://github.com/SmallDoges/small-doge/issues)
- 💡 **建议功能**: [讨论区](https://github.com/SmallDoges/small-doge/discussions)
- 📚 **改进文档**: 提交文档相关的PR
- 🏋️ **分享模型**: 贡献训练好的模型和配方
- 💬 **加入社区**: [Discord](https://discord.gg/P2yYH95N)

## 📚 文档

- **[📖 快速开始](./docs/quickstart.md)** - 5分钟上手指南
- **[⚙️ 安装说明](./docs/installation.md)** - 详细安装指南
- **[🎓 训练指南](./docs/training.md)** - 完整训练流程
- **[🤖 模型文档](./docs/models.md)** - 架构和性能详解
- **[🌐 Web界面](./docs/webui.md)** - 网页界面使用指南
- **[🔧 示例](./examples/)** - Jupyter笔记本和教程
- **[📊 评估](./evaluation/)** - 基准测试工具包

## 📄 引用

```bibtex
@misc{smalldoges2025,
    title={SmallDoges: A Family of Dynamic Ultra-Fast Small Language Models}, 
    author={Jingze Shi and Yifan Wu and Bingheng Wu and Yuyu Luo},
    year={2025},
    month={March},
    url={https://github.com/SmallDoges/small-doge}
}
```

## 📄 许可证

本项目采用Apache-2.0许可证 - 详见[LICENSE](LICENSE)文件。

---

<div align="center">

**由SmallDoge团队用❤️构建**

[![Star History](https://api.star-history.com/svg?repos=SmallDoges/small-doge&type=Date)](https://star-history.com/#SmallDoges/small-doge&Date)

*如果SmallDoge对您有帮助，请给我们一个⭐！*

</div>