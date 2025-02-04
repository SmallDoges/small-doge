# Doge训练指南

<div align="center">
<h4>

[English](https://github.com/SamllDoge/small-doge/blob/main/recipes/doge/README.md) | 简体中文

</h4>
</div>

*我们在本指南中提供了训练Doge的详细步骤, 包括预训练的Doge-Base, 指令微调的Doge-Instruct, 以及推理微调的Doge-R1.*

**目录**
1. [安装](#1-安装)
2. [预训练Base模型](#2-预训练Base模型)
    - [下载数据集](#21-下载数据集)
    - [预处理数据集](#22-预处理数据集)
    - [合并数据集](#23-合并数据集)
    - [配置模型参数](#24-配置模型参数)
    - [配置预训练超参数](#25-配置预训练超参数)
    - [预训练模型](#26-预训练模型)
    - [使用](#27-使用)
    - [评估](#28-评估)
3. [指令微调Instruct模型](#3-指令微调Instruct模型)
    - [下载数据集](#31-下载数据集)
    - [预处理数据集](#32-预处理数据集)
    - [合并数据集](#33-合并数据集)
    - [监督微调模型](#34-监督微调模型)
    - [直接偏好优化模型](#35-直接偏好优化模型)
    - [使用](#36-使用)
4. [推理微调R1模型](#4-推理微调R1模型)
    - [下载数据集](#41-下载数据集)
    - [预处理数据集](#42-预处理数据集)
    - [合并数据集](#43-合并数据集)
    - [蒸馏微调模型](#44-蒸馏微调模型)
    - [群体相对优化模型](#45-群体相对优化模型)
    - [使用](#46-使用)


## 1. 安装

请按照[README](https://github.com/SmallDoges/small-doge/blob/main/README_zh.md)中的说明安装必要的依赖项.


## 2. 预训练Base模型

我们提供了可以在新数据集上继续预训练的Doge检查点, 如果您有需要的话, 请参阅[这里](https://huggingface.co/collections/SmallDoge/doge-checkpoint-679ce95dae1d498e3ad35068)以获取更多信息.

### 2.1 下载数据集

预训练数据集, 我们选取了 `fineweb-edu-dedup` 高质量文本, `cosmopedia-v2` 合成指令数据集, 并补充 `python-edu` 与 `fine-math` 来保证模型的代码与数学能力. 

```shell
# 填写保存路径, 缓存路径和进程数
python ./examples/utils/download_pt_datasets.py --save_dir ./datasets --cache_dir ./cache --num_proc 1
```

> [!NOTE]
> 由于数据集过大, 至少需要 2TB 的存储空间. 如果您的存储空间不足, 可以自行在[这里](../../examples/utils/download_pt_datasets.py)选择部分数据集进行下载. 
> 您可以自由更改下载的数据集, 我们提供这个示例仅仅是为了复现目前的开源模型.

### 2.2 预处理数据集

我们需要使用 `tokenizer` 将数据集转为模型可接受的 `input_ids` 与 `attention_mask`.
如果使用 `LlamaTokenizer` , 该 tokenizer 词表大小为 `32768` , 使用 `[INST]` 与 `[/INST]` 标记指令. 它还包括工具标记, 但是我们不会在这里使用它们.
像 cosmopedia-v2 这样的数据集就包括 `prompt` 与 `text` 两个字段, 我们就将他们标记为用户内容与助手内容.

```python
conversation = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": text},
]
return tokenizer.apply_chat_template(conversation, tokenize=True, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_dict=True)
```

当然你也可以自行加入一些指令提示, 例如让模型能够回答我是Doge, 而不是 ChatGPT.

```python
conversation = [
    {"role": "user", "content": "你是谁?"},
    {"role": "assistant", "content": "我是一个名为 `Doge` 的 AI 助手, 我是由 `SmallDoge` 社区基于 `Doge` 架构训练的语言模型, 我的任务是根据用户的问题和请求提供适当的答案和支持."},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": text},
]
```

在这里我们推荐使用 [Doge-tokenizer](https://huggingface.co/SmallDoge/Doge-tokenizer) 来处理数据集, 它是由 `Llama-3.3` 的分词器针对 `smollm-corpus` 训练得到的, 词表大小为 `32768` , 训练脚本可以在 [这里](../../examples/utils/train_tokenizer_from_old.py) 找到.

```shell
# 填写数据集路径, 保存路径, 分词器路径, 样本数量, 最大长度和进程数
python ./examples/utils/preprocess_pt_datasets.py --datasets_dir ./datasets --save_dir ./datasets --tokenizer_name_or_path SamllDoge/Doge-tokenizer --train_examples 128000000 --test_examples 1000 --max_length 2048 --num_proc 16
```

> [!NOTE]
> 我们只保留 256B tokens 的数据集, 比例为 fineweb-edu:cosmopedia-v2:python-edu:open-web-math = 7:2:0.5:0.5, 如果你需要训练更大的模型, 请自行增加数据集的规模.

### 2.3 合并数据集

我们将 fineweb-edu_tokenized, cosmopedia-v2, python-edu 和 finemath 数据集合并为 `pretrain` 数据集.
然后将它们打乱顺序 `seed=233` , 并拆分出来 `1,000` 个样本作为测试集.

```shell
# 填写数据集路径, 保存路径, 样本数量和进程数
python ./examples/utils/concatenate_pt_datasets.py --datasets_dir ./datasets --save_dir ./datasets --train_examples 128000000 --test_examples 1000 --num_proc 16
```

### 2.4 配置模型参数

我们配置一个 `20M` 的小型模型, 进行训练测试.

| Model | Params | n_layers | d_model | d_ff | n_heads | kv_heads | n_exprets | n_expert_heads | n_expert_pre_head |
|---|---|---|---|---|---|---|---|---|---|
| Doge-20M | 13M | 8 | 256 | 512 | 2 | 1 | - | - | - |
| Doge-60M | 54M | 16 | 512 | 1024 | 4 | 2 | - | - | - |
| Doge-160M | 152M | 24 | 768 | 1536 | 6 | 3 | - | - | - |
| Doge-320M | 335M | 32 | 1024 | 2048 | 8 | 4 | - | - | - |

- n_layers 是模型的解码器层数
- d_model 是模型的隐藏层维度
- n_heads 是多头注意力头数 d_model // n_heads 最好保持在 64 以上

> [!TIP]
> `Doge-MoE` 模型可以继承 `Doge` 模型的密集激活参数, 并通过设置 `n_experts`, `n_expert_heads`, `n_expert_pre_head` 来增加稀疏激活的参数, 如果您希望增加模型参数而又不想增加计算成本, 可以尝试将模型配置的 `is_moe` 参数设置为 `True`, 并调整上述参数.

### 2.5 配置预训练超参数

| Model | tokens | max_train_steps | accumulate_steps | learning_rate | scheduler | warmup_ratio | decay_ratio | weight_decay | min_lr_rate |
|---|---|---|---|---|---|---|---|---|---|
| Doge-20M | 4B | 8,000 | 256 | 8e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |
| Doge-60M | 16B | 16,000 | 512 | 6e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |
| Doge-160M | 32B | 24,000 | 768 | 4e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |
| Doge-320M | 64B | 32,000 | 1024 | 2e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |

> [!TIP]
> 根据 [SmolLM博客](https://huggingface.co/blog/smollm) 的经验, 我们将 [Chinchilla](https://arxiv.org/pdf/2203.15556) 中参数与标记的缩放比例扩大 10 倍.
> 使用 `warmup_stable_decay` 是为了随时使用检查点在更大的数据集上继续训练, 参见 [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/pdf/2405.18392).

### 2.6 预训练模型

我们支持使用 Single GPU, DDP 或 DeepSpeed ZeRO-2 和 ZeRO-3 训练模型. 要在这四种方法之间切换, 只需更改 `recipes` 中 [`accelerate_configs`](../accelerate_configs) YAML 配置的路径.

> [!NOTE]
> 我们默认没有安装 `DeepSpeed`, 因为 Windows 系统不支持, 您如果需要使用, 请自行安装.

```shell
# 你需要指定配置文件路径, 所有参数都在配方配置文件中
ACCELERATE_LOG_LEVEL=info accelerate launch ./src/small_doge/pt.py --config_file recipes/accelerate_configs/single_gpu.yaml --config recipes/doge/Doge-20M/config_full.yaml
```

> [!NOTE]
> 上面的训练命令是为 1 x RTX 4090 (24GB) 节点配置的. 对于不同的硬件和拓扑, 您可能需要调整批量大小和梯度累积步数.

### 2.7 使用

在完成训练后, 我们可以使用 `Transformers` 的 `AutoModelForCausalLM` 加载模型, 并使用 `AutoTokenizer` 加载 `LlamaTokenizer` .

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-20M")
model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-20M", trust_remote_code=True)

inputs = tokenizer("Hey how are you doing?", return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.batch_decode(out))
```

### 2.8 评估

我们使用 [lighteval](https://github.com/huggingface/lighteval) 工具包来评估 Doge 模型的性能.

你可以使用以下命令安装工具包:

```bash
pip install lighteval
```

> [!NOTE]
> lighteval 默认会安装 `torch==2.4.1`, 也许你需要额外新建一个环境来安装.

如果您是 Linux 用户, 您可以使用以下命令来评估模型:

```bash
bash ./evaluation/eval_downstream_tasks.sh
```

如果您是 Windows 用户, 您可以使用以下命令来评估模型:

```bash
. ./evaluation/eval_downstream_tasks.ps1
```

> [!TIP]
> 您可以在脚本中修改 `MODEL` 和 `OUTPUT_DIR` 来评估不同的模型并将结果保存到不同的目录中.


## 3. 指令微调Instruct模型

我们提供了可以直接进行指令微调的基座Doge模型, 如果您有需要的话, 请参阅[这里](https://huggingface.co/collections/SmallDoge/doge-slm-679cc991f027c4a3abbded4a)以获取更多信息.

### 3.1 下载数据集

微调数据集, 我们选取了 `smoltalk` 数据集, 来进行监督微调, `ultrafeedback_binarized` 数据集, 来进行直接偏好优化.

```shell
# 填写保存路径, 缓存路径和进程数
python ./examples/utils/download_ft_dataset.py --save_dir ./datasets --cache_dir ./cache --num_proc 1
```

> [!TIP]
> 您可以自由更改下载的数据集, 我们提供这个示例仅仅是为了复现目前的开源模型.

### 3.2 预处理数据集

我们将微调数据集应用于`聊天模板`.

```python
# 填充数据集存放路径 数据集保存路径、分词器路径、进程数量 
python ./examples/utils/preprocess_ft_datasets.py --datasets_dir ./datasets --save_dir ./datasets --tokenizer_name_or_path SmallDoge/Doge-tokenizer --num_proc 8
```

> [!TIP]
> 您可以在模板中添加一些指令提示, 例如让模型能够回答我是Doge, 而不是 ChatGPT.

### 3.3 合并数据集

如果你下载了更多数据集进行微调, 我们需要将其合并打乱来混合在一起.

```shell
# 填写数据集路径, 保存路径, 样本数量和进程数
python ./examples/utils/concatenate_ft_datasets.py --datasets_dir ./datasets --save_dir ./datasets --num_proc 16
```

### 3.4 监督微调模型

我们首先对模型进行监督微调, 使其更够跟随 `prompt` 来生成回复.

```shell
# 你需要指定配置文件路径, 所有参数都在配方配置文件中
ACCELERATE_LOG_LEVEL=info accelerate launch ./src/small_doge/sft.py --config_file recipes/accelerate_configs/single_gpu.yaml --config recipes/doge/Doge-20M-Instruct/sft/config_full.yaml
```

> [!NOTE]
> 上面的训练命令是为 1 x RTX 4090 (24GB) 节点配置的. 对于不同的硬件和拓扑, 您可能需要调整批量大小和梯度累积步数.

### 3.5 直接偏好优化模型

然后我们对监督微调后的模型进行强化学习, 来与人类偏好对齐, 这里使用的是 `DPO` 算法.

```shell
# 你需要指定配置文件路径, 所有参数都在配方配置文件中
ACCELERATE_LOG_LEVEL=info accelerate launch ./src/small_doge/dpo.py --config_file recipes/accelerate_configs/single_gpu.yaml --config recipes/doge/Doge-20M-Instruct/dpo/config_full.yaml
```

> [!NOTE]
> 上面的训练命令是为 1 x RTX 4090 (24GB) 节点配置的. 对于不同的硬件和拓扑, 您可能需要调整批量大小和梯度累积步数.

### 3.6 使用

在完成微调后, 我们可以使用 `Transformers` 的 `AutoModelForCausalLM` 加载模型, 使用 `AutoTokenizer` 加载 `LlamaTokenizer`, 并使用 `GenerationConfig` 和 `TextStreamer` 来支持带有采样的流式生成.

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
steamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

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

## 4. 推理微调R1模型

目前可以进行推理微调的教师模型蒸馏数据仍然比较稀少, 这里我们提供huggingface的[open-r1](https://github.com/huggingface/open-r1/?tab=readme-ov-file#data-generation)项目链接, 如果您有需要的话, 可以自行根据指南使用OpenAI的o1或DeepSeek的R1模型来生成教师模型数据.

### 4.1 下载数据集

微调数据集, 我们选取了 `Bespoke-Stratos-17k` 数据集, 来进行蒸馏微调, `NuminaMath-TIR` 数据集, 来进行群体相对优化.

```shell
# 填写保存路径, 缓存路径和进程数
python ./examples/utils/download_ft_dataset.py --save_dir ./datasets --cache_dir ./cache --num_proc 1
```

> [!NOTE]
> 如果你完成了 `指令微调Instruct模型` 指南, 并且没有更改下载数据集脚本或删除数据集, 那么你可以跳过这一步.
> [!TIP]
> 您可以自由更改下载的数据集.

### 4.2 预处理数据集

我们为微调数据集应用 `思考提示` .

```python
# 填充数据集存放路径 数据集保存路径、分词器路径、进程数量
python ./examples/utils/preprocess_ft_datasets.py --datasets_dir ./datasets --save_dir ./datasets --tokenizer_name_or_path SmallDoge/Doge-tokenizer --num_proc 8
```

> [!NOTE]
> 如果你完成了 `指令微调Instruct模型` 指南, 并且没有更改预处理数据集脚本或删除数据集, 那么你可以跳过这一步.
> [!TIP]
> 您可以在思考提示中额外添加一些行为指令, 来构建更有趣的对话.

### 4.3 合并数据集

如果你下载了更多数据集进行微调, 我们需要将其合并打乱来混合在一起.

```shell
# 填写数据集路径, 保存路径, 样本数量和进程数
python ./examples/utils/concatenate_ft_datasets.py --datasets_dir ./datasets --save_dir ./datasets --num_proc 16
```

> [!NOTE]
> 如果你完成了 `指令微调Instruct模型` 指南, 并且没有更改合并数据集脚本或删除数据集, 那么你可以跳过这一步.

### 4.4 蒸馏微调模型

我们首先对模型进行蒸馏微调, 从教师模型学习到强大的思维推理能力.

```shell
# 你需要指定配置文件路径, 所有参数都在配方配置文件中
ACCELERATE_LOG_LEVEL=info accelerate launch ./src/small_doge/sft.py --config_file recipes/accelerate_configs/single_gpu.yaml --config recipes/doge/Doge-20M-R1/sft/config_full.yaml
```

> [!NOTE]
> 上面的训练命令是为 1 x RTX 4090 (24GB) 节点配置的. 对于不同的硬件和拓扑, 您可能需要调整批量大小和梯度累积步数.

### 4.5 群体相对优化模型

然后我们对蒸馏微调后的模型进行强化学习, 来使模型具备回答前进行思考的能力, 这里使用的是 `GRPO` 算法.

```shell
# 你需要指定配置文件路径, 所有参数都在配方配置文件中
ACCELERATE_LOG_LEVEL=info accelerate launch ./src/small_doge/grpo.py --config_file recipes/accelerate_configs/single_gpu.yaml --config recipes/doge/Doge-20M-R1/grpo/config_full.yaml
```

> [!NOTE]
> 上面的训练命令是为 1 x RTX 4090 (24GB) 节点配置的. 对于不同的硬件和拓扑, 您可能需要调整批量大小和梯度累积步数.

### 4.6 使用

在完成微调后, 我们可以使用 `Transformers` 的 `AutoModelForCausalLM` 加载模型, 使用 `AutoTokenizer` 加载 `LlamaTokenizer`, 并使用 `GenerationConfig` 和 `TextStreamer` 来支持带有采样的流式生成.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-20M-R1")
model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-20M-R1", trust_remote_code=True)

generation_config = GenerationConfig(
      max_new_tokens=1000, 
      use_cache=True, 
      do_sample=True, 
      temperature=0.8, 
      top_p=0.9,
      repetition_penalty=1.0
)
steamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

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
