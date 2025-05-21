# Copyright 2025 The SmallDoge Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import re
import json

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainerCallback
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from tqdm import tqdm

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # 设置保存最佳模型的参数
    if training_args.eval_strategy != "no" or training_args.evaluation_strategy != "no":
        training_args.load_best_model_at_end = True 
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
        training_args.save_total_limit = 3  # 只保留最好的3个checkpoint

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    # if re.match(r'^[^/]+/[^/]+$', script_args.dataset_name):
    # dataset = load_dataset(
    #     'parquet',
    #     data_files={
    #         script_args.dataset_name + "/*.parquet"
    #     },
    #     split='train'
    #     # load_from_cache_file=False
    # )
    # else:
    dataset = load_from_disk(script_args.dataset_name)
    if "test" not in dataset:
        dataset = dataset.train_test_split(test_size=1000)
    dataset["train"] = dataset["train"]
    
    def preprocess_function(examples):
        messages = []
        short_system_prompt = "As an assistant, you need to thoroughly explore the problem through precise thinking process before providing the final accurate solution. The thinking process includes Analysis, First, Second, Next, Reflection, Finally and Summarizing behavioral steps to develop a well-considered thought process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {**Analysis:**\\n\\n**First:**\\n\\n**Second:**\\n\\n**Next:**\\n\\n**Reflection:**\\n\\n**Finally:**\\n\\n**Summarizing:**} <|end_of_thought|>. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {**Solution:**} <|end_of_solution|>."
        messages.append({"role": "system", "content": short_system_prompt})
        messages.append(examples["messages"][0])
        messages.append(examples["messages"][1])
        return {"messages": messages}
    
    dataset = dataset.map(preprocess_function)

    for split in dataset:
        dataset[split] = dataset[split].remove_columns([col for col in dataset[split].column_names if col != "messages"])
    # print(dataset["train"])
    print(dataset["train"][0])

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    
    ############################
    # Apply Liger Kernel
    ############################
    apply_liger_kernel_to_qwen2(
        rope=True,
        rms_norm=True,
        swiglu=True,
        cross_entropy=True,
        fused_linear_cross_entropy=False,
)

    ############################
    # Initialize the SFT Trainer
    ############################
    # metrics_callback = MetricsCallback(training_args.output_dir)
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        # callbacks=[metrics_callback]
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Start training... ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Saving model... ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "tags": ["small-doge"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    logger.info("*** Training complete ***")
    # metrics_callback.save_metrics()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Start evaluation... ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("*** Evaluation complete ***")

    # #############
    # # push to hub
    # #############
    # if training_args.push_to_hub:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)

    logger.info("*** Training finished! ***")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
