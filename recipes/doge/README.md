# Instructions to train Doge

<div align="center">
<h4>

English | [简体中文](https://github.com/SamllDoge/small-doge/blob/main/recipes/doge/README_zh.md)

</h4>
</div>

*We provide detailed steps to train Doge in this guide, including pre-training Doge-Base, instruction fine-tuning Doge-Instruct, and reasoning fine-tuning Doge-R1.*

**Table of Contents**
1. [Installation](#1-installation)
2. [Pre-training Base model](#2-pre-training-base-model)
    - [Download the dataset](#21-download-the-dataset)
    - [Preprocess the dataset](#22-preprocess-the-dataset)
    - [Concatenate the dataset](#23-concatenate-the-dataset)
    - [Configure the model parameters](#24-configure-the-model-parameters)
    - [Configure the pre-training hyperparameters](#25-configure-the-pre-training-hyperparameters)
    - [Pre-training the model](#26-pre-training-the-model)
    - [Usage](#27-usage)
    - [Evaluation](#28-evaluation)
3. [Instruction Fine-tuning Instruct model](#3-instruction-fine-tuning-instruct-model)
    - [Download the dataset](#31-download-the-dataset)
    - [Preprocess the dataset](#32-preprocess-the-dataset)
    - [Concatenate the dataset](#33-concatenate-the-dataset)
    - [Supervised Fine-tuning the model](#34-supervised-fine-tuning-the-model)
    - [Direct Preference Optimization the model](#35-direct-preference-optimization-the-model)
    - [Usage](#36-usage)
4. [Reasoning Fine-tuning R1 model](#4-reasoning-fine-tuning-r1-model)
    - [Download the dataset](#41-download-the-dataset)
    - [Preprocess the dataset](#42-preprocess-the-dataset)
    - [Concatenate the dataset](#43-concatenate-the-dataset)
    - [Distillation Fine-tuning the model](#44-distillation-fine-tuning-the-model)
    - [Group Relative Optimization the model](#45-group-relative-optimization-the-model)
    - [Usage](#46-usage)


## 1. Installation

Please follow the instructions in [README](https://github.com/SmallDoges/small-doge/blob/main/README.md) to install the necessary dependencies.


## 2. Pre-training Base model

We provide a Doge checkpoint that can be further pre-trained on a new dataset. If you need it, please refer to [here](https://huggingface.co/collections/SmallDoge/doge-checkpoint-679ce95dae1d498e3ad35068) for more information.

### 2.1 Download the dataset

For the pre-training dataset, we selected the `fineweb-edu-dedup` high-quality text, `cosmopedia-v2` synthetic instruction dataset, and supplemented `python-edu` and `fine-math` to ensure the model's code and math capabilities.

```shell
# Fill in the save path, cache path, and number of processes. (Optional, if you want to accelerate datasets installed speed change parallel option to True)
python ./examples/utils/download_pt_datasets.py --save_dir ./datasets --cache_dir ./cache --num_proc 1 --is_parallel False
```

> [!NOTE]
> Due to the large size of the dataset, at least 2TB of storage space is required. If you do not have enough storage space, you can choose to download part of the dataset by yourself [here](../../examples/utils/download_pt_datasets.py).
> You can freely change the downloaded dataset. We provide this example just to reproduce the current open-source model.

### 2.2 Preprocess the dataset

We need to use the `tokenizer` to convert the dataset into `input_ids` and `attention_mask` that the model can accept.
If you use `LlamaTokenizer`, the tokenizer's vocabulary size is `32768`, and it uses `[INST]` and `[/INST]` to mark instructions. It also includes tool tokens, but we will not use them here.
Datasets like cosmopedia-v2 include two fields, `prompt` and `text`, which we mark as user content and assistant content.

```python
conversation = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": text},
]
return tokenizer.apply_chat_template(conversation, tokenize=True, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_dict=True)
```

Of course, you can also add some instruction prompts by yourself, such as letting the model answer that it is Doge, not ChatGPT.

```python
conversation = [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am an AI assistant named `Doge`. I am a language model trained by the `SmallDoge` community based on the `Doge` architecture. My task is to provide appropriate answers and support based on the user's questions and requests."},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": text},
]
```

Here we recommend using [Doge-tokenizer](https://huggingface.co/SmallDoge/Doge-tokenizer) to process the dataset. It is trained by the `Llama-3.3` tokenizer on the `smollm-corpus`, with a vocabulary size of `32768`. The training script can be found [here](../../examples/utils/train_tokenizer_from_old.py).

```shell
# Fill in the dataset path, save path, tokenizer path, sample number, maximum length, and number of processes
python ./examples/utils/preprocess_pt_datasets.py --datasets_dir ./datasets --save_dir ./datasets --tokenizer_name_or_path SamllDoge/Doge-tokenizer --train_examples 128000000 --test_examples 1000 --max_length 2048 --num_proc 16
```

> [!NOTE]
> We only keep the dataset with 256B tokens, and the ratio of fineweb-edu:cosmopedia-v2:python-edu:open-web-math = 7:2:0.5:0.5. If you need to train a larger model, please increase the scale of the dataset by yourself.

### 2.3 Concatenate the dataset

We concatenate the fineweb-edu_tokenized, cosmopedia-v2, python-edu, and finemath datasets into the `pretrain` dataset.
Then we shuffle them in order `seed=233`, and split out `1,000` samples as the test set.

```shell
# Fill in the dataset path, save path, sample number, and number of processes
python ./examples/utils/concatenate_pt_datasets.py --datasets_dir ./datasets --save_dir ./datasets --train_examples 128000000 --test_examples 1000 --num_proc 16
```

### 2.4 Configure the model parameters

We configure a `20M` small model for training and testing.

| Model | Params | n_layers | d_model | d_ff | n_heads | kv_heads | n_exprets | n_expert_heads | n_expert_pre_head |
|---|---|---|---|---|---|---|---|---|---|
| Doge-20M | 13M | 8 | 256 | 512 | 2 | 1 | - | - | - |
| Doge-60M | 54M | 16 | 512 | 1024 | 4 | 2 | - | - | - |
| Doge-160M | 152M | 24 | 768 | 1536 | 6 | 3 | - | - | - |
| Doge-320M | 335M | 32 | 1024 | 2048 | 8 | 4 | - | - | - |

- n_layers is the number of decoder layers in the model
- d_model is the hidden layer dimension of the model
- n_heads is the number of heads of multi-head attention, d_model // n_heads is best kept above 64

> [!TIP]
> The `Doge-MoE` model can inherit the dense activation parameters of the `Doge` model and increase the sparse activation parameters by setting `n_experts`, `n_expert_heads`, and `n_expert_pre_head`. If you want to increase the model parameters without increasing the computational cost, you can try setting the `is_moe` parameter of the model configuration to `True` and adjust the above parameters.

### 2.5 Configure the pre-training hyperparameters

| Model | tokens | max_train_steps | accumulate_steps | learning_rate | scheduler | warmup_ratio | decay_ratio | weight_decay | min_lr_rate |
|---|---|---|---|---|---|---|---|---|---|
| Doge-20M | 4B | 8,000 | 256 | 8e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |
| Doge-60M | 16B | 16,000 | 512 | 6e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |
| Doge-160M | 32B | 24,000 | 768 | 4e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |
| Doge-320M | 64B | 32,000 | 1024 | 2e-3 | warmup_stable_decay | 0.1 | 0.1 | 0.01 | 0.0 |

> [!TIP]
> According to the experience of [SmolLM blog](https://huggingface.co/blog/smollm), we will scale the parameters in [Chinchilla](https://arxiv.org/pdf/2203.15556) by 10 times the scaling ratio of tokens.
> `warmup_stable_decay` is used to continue training with checkpoints on larger datasets at any time, see [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/pdf/2405.18392).

### 2.6 Pre-training the model

We support training the model using Single GPU, DDP, or DeepSpeed ZeRO-2 and ZeRO-3. To switch between these four methods, simply change the path in the [`accelerate_configs`](../accelerate_configs) YAML configuration in the `recipes` directory.

> [!NOTE]
> We do not install `DeepSpeed` by default because Windows systems do not support it. If you need to use it, please install it yourself.

```shell
# You need to specify the configuration file path, all parameters are in the recipe configuration file
ACCELERATE_LOG_LEVEL=info accelerate launch ./src/small_doge/pt.py --config_file recipes/accelerate_configs/single_gpu.yaml --config recipes/doge/Doge-20M/config_full.yaml
```

> [!NOTE]
> The training command above is configured for a 1 x RTX 4090 (24GB) node. For different hardware and topologies, you may need to adjust the batch size and gradient accumulation steps.

### 2.7 Usage

After training is complete, we can use `AutoModelForCausalLM` of `Transformers` to load the model, and use `AutoTokenizer` to load `LlamaTokenizer`.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-20M")
model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-20M", trust_remote_code=True)

inputs = tokenizer("Hey how are you doing?", return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.batch_decode(out))
```

### 2.8 Evaluation

We use the [lighteval](https://github.com/huggingface/lighteval) toolkit to evaluate the performance of the Doge model.

You can install the toolkit with the following command:

```bash
pip install lighteval
```

> [!NOTE]
> By default, lighteval installs `torch==2.4.1`, so you may need to create a new environment to install it.

If you are a Linux user, you can use the following command to evaluate the model:

```bash
bash ./evaluation/eval_downstream_tasks.sh
```

If you are a Windows user, you can use the following command to evaluate the model:

```powershell
. ./evaluation/eval_downstream_tasks.ps1
```

> [!TIP]
> You can modify `MODEL` and `OUTPUT_DIR` in the script to evaluate different models and save the results to different directories.


## 3. Instruction Fine-tuning Instruct model

We provide a base Doge model that can be fine-tuned directly for instruction fine-tuning. If you need it, please refer to [here](https://huggingface.co/collections/SmallDoge/doge-slm-679cc991f027c4a3abbded4a) for more information.

### 3.1 Download the dataset

For the fine-tuning dataset, we selected the `smoltalk` dataset for SFT and the `ultrafeedback_binarized` dataset for DPO.

```shell
# Fill in the save path, cache path, and number of processes
python ./examples/utils/download_ft_dataset.py --save_dir ./datasets --cache_dir ./cache --num_proc 1
```

> [!TIP]
> You can freely change the downloaded dataset. We provide this example just to reproduce the current open-source model.

### 3.2 Preprocess the dataset

We apply the fine-tuning dataset to the `chat template`.

```python
# Fill in the dataset path, save path, tokenizer path, and number of processes
python ./examples/utils/preprocess_ft_datasets.py --datasets_dir ./datasets --save_dir ./datasets --tokenizer_name_or_path SmallDoge/Doge-tokenizer --num_proc 8
```

> [!TIP]
> You can add some instruction prompts in the template by yourself, such as letting the model answer that it is Doge, not ChatGPT.

### 3.3 Concatenate the dataset

If you download more datasets for fine-tuning, we need to concatenate and shuffle them to mix them together.

```shell
# Fill in the dataset path, save path, sample number, and number of processes
python ./examples/utils/concatenate_ft_datasets.py --datasets_dir ./datasets --save_dir ./datasets --num_proc 16
```

### 3.4 Supervised Fine-tuning the model

We first SFT the model to make it generate responses that follow the `prompt`.

```shell
# You need to specify the configuration file path, all parameters are in the recipe configuration file
ACCELERATE_LOG_LEVEL=info accelerate launch ./src/small_doge/sft.py --config_file recipes/accelerate_configs/single_gpu.yaml --config recipes/doge/Doge-20M-Instruct/sft/config_full.yaml
```

> [!NOTE]
> The training command above is configured for a 1 x RTX 4090 (24GB) node. For different hardware and topologies, you may need to adjust the batch size and gradient accumulation steps.

### 3.5 Direct Preference Optimization the model

Then we use the DPO algorithm to align the model with human preferences after SFT.

```shell
# You need to specify the configuration file path, all parameters are in the recipe configuration file
ACCELERATE_LOG_LEVEL=info accelerate launch ./src/small_doge/dpo.py --config_file recipes/accelerate_configs/single_gpu.yaml --config recipes/doge/Doge-20M-Instruct/dpo/config_full.yaml
```

> [!NOTE]
> The training command above is configured for a 1 x RTX 4090 (24GB) node. For different hardware and topologies, you may need to adjust the batch size and gradient accumulation steps.

### 3.6 Usage

After fine-tuning is complete, we can use `AutoModelForCausalLM` of `Transformers` to load the model, and use `AutoTokenizer` to load `LlamaTokenizer`, and use `GenerationConfig` and `TextStreamer` to support streaming generation with sampling.

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


## 4. Reasoning Fine-tuning R1 model

Currently, the data for reasoning fine-tuning the teacher model is still relatively scarce. Here we provide the huggingface [open-r1](https://github.com/huggingface/open-r1/?tab=readme-ov-file#data-generation) project link. If you need it, you can use OpenAI's o1 or DeepSeek's R1 model to generate teacher model data according to the guide.

### 4.1 Download the dataset

For the fine-tuning dataset, we selected the `Bespoke-Stratos-17k` dataset for DFT and the `NuminaMath-TIR` dataset for GRPO.

```shell
# Fill in the save path, cache path, and number of processes
python ./examples/utils/download_ft_dataset.py --save_dir ./datasets --cache_dir ./cache --num_proc 1
```

> [!NOTE]
> If you have completed the `Instruction Fine-tuning Instruct model` guide and have not changed the download dataset script or deleted the dataset, you can skip this step.
> [!TIP]
> You can freely change the downloaded dataset.

### 4.2 Preprocess the dataset

We apply the fine-tuning dataset to the `thinking prompt`.

```python
# Fill in the dataset path, save path, tokenizer path, and number of processes
python ./examples/utils/preprocess_ft_datasets.py --datasets_dir ./datasets --save_dir ./datasets --tokenizer_name_or_path SmallDoge/Doge-tokenizer --num_proc 8
```

> [!NOTE]
> If you have completed the `Instruction Fine-tuning Instruct model` guide and have not changed the preprocess dataset script or deleted the dataset, you can skip this step.
> [!TIP]
> You can add some behavior instructions in the thinking prompt by yourself to build more interesting conversations.

### 4.3 Concatenate the dataset

If you download more datasets for fine-tuning, we need to concatenate and shuffle them to mix them together.

```shell
# Fill in the dataset path, save path, sample number, and number of processes
python ./examples/utils/concatenate_ft_datasets.py --datasets_dir ./datasets --save_dir ./datasets --num_proc 16
```

> [!NOTE]
> If you have completed the `Instruction Fine-tuning Instruct model` guide and have not changed the concatenate dataset script or deleted the dataset, you can skip this step.

### 4.4 Distillation Fine-tuning the model

We first DFT the model to learn powerful thinking and reasoning capabilities from the teacher model.

```shell
# You need to specify the configuration file path, all parameters are in the recipe configuration file
ACCELERATE_LOG_LEVEL=info accelerate launch ./src/small_doge/sft.py --config_file recipes/accelerate_configs/single_gpu.yaml --config recipes/doge/Doge-20M-R1/sft/config_full.yaml
```

> [!NOTE]
> The training command above is configured for a 1 x RTX 4090 (24GB) node. For different hardware and topologies, you may need to adjust the batch size and gradient accumulation steps.

### 4.5 Group Relative Optimization the model

Then we use the GRPO algorithm to reinforce the model after DFT to make the model have the ability to think before answering, which is the `GRPO` algorithm.

```shell
# You need to specify the configuration file path, all parameters are in the recipe configuration file
ACCELERATE_LOG_LEVEL=info accelerate launch ./src/small_doge/grpo.py --config_file recipes/accelerate_configs/single_gpu.yaml --config recipes/doge/Doge-20M-R1/grpo/config_full.yaml
```

> [!NOTE]
> The training command above is configured for a 1 x RTX 4090 (24GB) node. For different hardware and topologies, you may need to adjust the batch size and gradient accumulation steps.

### 4.6 Usage

After fine-tuning is complete, we can use `AutoModelForCausalLM` of `Transformers` to load the model, and use `AutoTokenizer` to load `LlamaTokenizer`, and use `GenerationConfig` and `TextStreamer` to support streaming generation with sampling.

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
