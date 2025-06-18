---
library_name: transformers
license: apache-2.0
datasets:
- HuggingFaceTB/smollm-corpus
language:
- en
pipeline_tag: text-generation
---

# **Doge 160M checkpoint**

![wsd_scheduler](./wsd_scheduler.png)

Doge uses `wsd_scheduler` as the training scheduler, which divides the learning rate into three stages: `warmup`, `stable`, and `decay`. It allows us to continue training on any new dataset from any checkpoint in the `stable stage` without spikes of the training.

Here are the initial learning rates required to continue training at each checkpoint:

- **[Doge-20M](https://huggingface.co/SmallDoge/Doge-20M-checkpoint)**: 8e-3
- **[Doge-60M](https://huggingface.co/SmallDoge/Doge-60M-checkpoint)**: 6e-3
- **[Doge-160M](https://huggingface.co/SmallDoge/Doge-160M-checkpoint)**: 4e-3
- **[Doge-320M](https://huggingface.co/SmallDoge/Doge-320M-checkpoint)**: 2e-3

| Model | Learning Rate | Schedule | Warmup Steps | Stable Steps |
|-------|---------------|----------|--------------|--------------|
| [Doge-20M](https://huggingface.co/SmallDoge/Doge-20M-checkpoint) | 8e-3 | wsd_scheduler | 800 | 6400 |
| [Doge-60M](https://huggingface.co/SmallDoge/Doge-60M-checkpoint) | 6e-3 | wsd_scheduler | 1600 | 12800 |
| [Doge-160M](https://huggingface.co/SmallDoge/Doge-160M-checkpoint) | 4e-3 | wsd_scheduler | 2400 | 19200 |
| [Doge-320M](https://huggingface.co/SmallDoge/Doge-320M-checkpoint) | 2e-3 | wsd_scheduler | 3200 | 25600 |
