# Instructions to evaluate Doge

We use the [lighteval](https://github.com/huggingface/lighteval) toolkit to evaluate the performance of the Doge model.

## Setup

You can install the toolkit using the following command:

```bash
pip install lighteval
```

## Evaluation

If you are a Linux user, you can use the following command to evaluate the model:

```bash
bash ./evaluation/eval_downstream_tasks.sh
```

If you are a Windows user, you can use the following command to evaluate the model:

```bash
. ./evaluation/eval_downstream_tasks.ps1
```

> [!TIP]
> You can modify `MODEL` and `OUTPUT_DIR` in the script to evaluate different models and save the results to different directories.