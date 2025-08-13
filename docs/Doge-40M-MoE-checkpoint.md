---
library_name: transformers
license: apache-2.0
datasets:
- SmallDoge/SmallCorpus
language:
- en
- zh
pipeline_tag: text-generation
---

# **Doge 40M MoE checkpoint**

Doge uses `wsd_scheduler` as the training scheduler, which divides the learning rate into three stages: `warmup`, `stable`, and `decay`. It allows us to continue training on any new dataset from any checkpoint in the `stable stage` without spikes in training.

Here are the initial learning rates required to continue training at each checkpoint:

- [Doge-40M](https://huggingface.co/SmallDoge/Doge-40M-checkpoint): 8e-3
- **[Doge-40M-MoE](https://huggingface.co/SmallDoge/Doge-40M-MoE-checkpoint): 8e-3**


| Model | Learning Rate | Schedule | Warmup Steps | Stable Steps |
|-------|---------------|----------|--------------|--------------|
| [Doge-40M](https://huggingface.co/SmallDoge/Doge-40M-checkpoint) | 8e-3 | wsd_scheduler | 2000 | 4000 |
| [Doge-40M-MoE](https://huggingface.co/SmallDoge/Doge-40M-MoE-checkpoint) | 8e-3 | wsd_scheduler | 2000 | 4000 |
