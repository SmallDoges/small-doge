---
library_name: transformers
license: apache-2.0
datasets:
- open-thoughts/OpenThoughts-114k
- open-r1/OpenR1-Math-220k
- SmallDoge/Reason-Distill
base_model:
- SmallDoge/Doge-160M
language:
- en
pipeline_tag: question-answering
tags:
- trl
- sft
- grpo
- doge
---


# **Doge 160M Reason Distill**

<div align="center">
  <img src="https://huggingface.co/spaces/SmallDoge/README/resolve/main/org_icon.png" width="100%" alt="SmallDoge" />
</div>
<hr>
<div align="center">
  <a href="https://discord.gg/P2yYH95N" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-Small%20Doges-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2412.11834" target="_blank" style="margin: 2px;">
    <img alt="arXiv" src="https://img.shields.io/static/v1?label=arXiv&message=2412.11834&color=B31B1B&logo=arXiv" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/SmallDoges/small-doge" target="_blank" style="margin: 2px;">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-SmallDoge-181717?logo=github" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/SmallDoges/small-doge/blob/main/LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

Doge uses Dynamic Mask Attention as sequence transformation and can use Multi-Layer Perceptron or Cross Domain Mixture of Experts as state transformation. Dynamic Mask Attention allows the Transformer to use self-attention during training and state space during inference, and Cross Domain Mixture of Experts can directly inherit the weights of Multi-Layer Perceptron for further training. This model is trained by [SmallDoge](https://huggingface.co/SmallDoge) community, for detailed algorithm and model architecture, please refer to [Wonderful Matrices](https://arxiv.org/abs/2412.11834), all training details and code are publicly available on the [small-doge](https://github.com/SmallDoges/small-doge) repository.


## Uses

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-160M-Reason-Distill")
model = AutoModelForCausalLM.from_pretrained("SmallDoge/Doge-160M-Reason-Distill", trust_remote_code=True)

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

system_prompt = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:
""".strip()
prompt = "Which number is bigger, 3.9 or 3.11?"
conversation = [
    {"role": "system", "content": system_prompt},
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

We build the Doge-Reason-Distill by SFT on [Reason-Distill](https://huggingface.co/datasets/SmallDoge/Reason-Distill).

> TODO: The larger model is under training and will be uploaded soon.

**SFT**:
| Model | Training Data | Epochs | Content Length | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|
| [Doge-160M-Reason-Distil](https://huggingface.co/SmallDoge/Doge-160M-Reason-Distill) | [SmallDoge/Reason-Distill](https://huggingface.co/datasets/SmallDoge/Reason-Distill) | 2 | 4096 | 4e-4 | 0.5M | bfloat16 |


**Procedure**:

**SFT**:
[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/loser_cheems/huggingface/runs/zgk4eefz) 


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