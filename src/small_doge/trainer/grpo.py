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
import re
import sys

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from small_doge.utils import (
    get_modeling_classes,
    register_model_classes,
    GRPOConfig,
    GRPOScriptArguments,
    REWARD_FUNCS_REGISTRY,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
)
from small_doge.processor import mix_grpo_datasets
from trl import (
    ModelConfig,
    GRPOTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.getLogger(__name__)


def main(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
):
    # Set seed for reproducibility
    set_seed(training_args.seed)

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
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Recipe type: {training_args.recipe_type}")

    ######################
    # Determine model type
    ######################
    recipe_type = training_args.recipe_type.lower()
    is_doge2 = recipe_type == 'doge2'
    logger.info(f"Using {'Doge2' if is_doge2 else 'Doge'} model")
    
    # Get model classes
    config_class, model_class, causal_lm_class = get_modeling_classes(recipe_type)

    ###############
    # Load datasets
    ###############
    logger.info("Using processor for dataset mixing and processing")
    dataset = mix_grpo_datasets(
        datasets_and_ratios=training_args.datasets_and_ratios,
        total_sample_size=training_args.total_sample_size,
        dataset_config=script_args.dataset_config,
        system_prompt=training_args.system_prompt,
    )


    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    if training_args.datasets_and_ratios:
        # Use dataset mixing
        logger.info("Using dataset mixing for GRPO training")
        dataset = mix_grpo_datasets(
            datasets_and_ratios=training_args.datasets_and_ratios,
            total_sample_size=training_args.total_sample_size,
            processing_class=tokenizer,
            formatting_func=None,
            dataset_num_proc=training_args.dataset_num_proc if hasattr(training_args, 'dataset_num_proc') else 4,
            seed=42,
            cache_dir=training_args.cache_dir if hasattr(training_args, 'cache_dir') else None,
            tools=training_args.tools if hasattr(training_args, 'tools') else None,
        )
        
        # Split dataset for train/eval
        if training_args.eval_strategy != "no":
            train_dataset = dataset["train"] if "train" in dataset else dataset
            eval_dataset = dataset["test"] if "test" in dataset else None
            if eval_dataset is None and "validation" in dataset:
                eval_dataset = dataset["validation"]
        else:
            train_dataset = dataset["train"] if "train" in dataset else dataset
            eval_dataset = None
    else:
        # Load single dataset
        if re.match(r'^[^/]+/[^/]+$', script_args.dataset_name):
            dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        else:
            dataset = load_from_disk(script_args.dataset_name)
        
        # Preprocess function
        def preprocess_function(example):
            prompt = []
            if training_args.system_prompt is not None:
                prompt.append({"role": "system", "content": training_args.system_prompt})
            prompt.append({"role": "user", "content": example["problem"]})
            
            processed = {"prompt": prompt}
            
            # Keep solution if available for reward calculation
            if "solution" in example:
                processed["solution"] = example["solution"]
                
            return processed

        dataset = dataset.map(preprocess_function)

        # Remove unnecessary columns
        for split in dataset:
            if "messages" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("messages")

        train_dataset = dataset[script_args.dataset_train_split]
        eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    # Get reward functions
    reward_funcs = []
    for func_name in script_args.reward_funcs:
        if func_name == "cosine":
            reward_func = get_cosine_scaled_reward(
                min_value_wrong=script_args.cosine_min_value_wrong,
                max_value_wrong=script_args.cosine_max_value_wrong,
                min_value_correct=script_args.cosine_min_value_correct,
                max_value_correct=script_args.cosine_max_value_correct,
                max_len=script_args.cosine_max_len,
            )
        elif func_name == "repetition_penalty":
            reward_func = get_repetition_penalty_reward(
                ngram_size=script_args.repetition_n_grams,
                max_penalty=script_args.repetition_max_penalty,
            )
        else:
            reward_func = REWARD_FUNCS_REGISTRY[func_name]
        reward_funcs.append(reward_func)

    logger.info(f"Using reward functions: {script_args.reward_funcs}")

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if model_args.quantization_config is not None else None,
        quantization_config=get_quantization_config(model_args),
    )
    training_args.model_init_kwargs = model_kwargs

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    else:
        logger.info("No checkpoint found, starting training from scratch.")

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
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
    metrics["train_samples"] = len(train_dataset)
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
        "tags": ["small-doge", "grpo"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    logger.info("*** Training complete ***")

    ##########
    # Evaluate
    ##########
    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Start evaluation... ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("*** Evaluation complete ***")

    ################################
    # Register and save model
    ################################
    register_model_classes(recipe_type)
    
    # Save tokenizer and model with proper class registration
    tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    model = AutoModelForCausalLM.from_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)

    #############
    # Push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** GRPO Training finished! ***")


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
