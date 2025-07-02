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
from argparse import ArgumentParser

import yaml
import datasets
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from small_doge.utils import (
    get_modeling_classes, 
    register_model_classes,
    DogeWarmupCallback,
    Doge2WarmupCallback,
    PTConfig,
)
from small_doge.processor import mix_pt_datasets
from trl import ModelConfig, ScriptArguments, TrlParser


logger = logging.getLogger(__name__)


def main(
    script_args: ScriptArguments,
    training_args: PTConfig,
    model_args: ModelConfig,
    model_config: dict,
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
    logger.info(f"Data parameters {training_args}")

    ######################
    # Determine model type
    ######################
    recipe_type = training_args.recipe_type.lower()
    is_doge2 = recipe_type == 'doge2'
    logger.info(f"Using {'Doge2' if is_doge2 else 'Doge'} model")

    # Get model classes
    config_class, model_class, causal_lm_class = get_modeling_classes(recipe_type)

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    logger.info("Using processor for dataset mixing and processing")
    dataset = mix_pt_datasets(
        datasets_and_ratios=training_args.datasets_and_ratios,
        total_sample_size=training_args.total_sample_size,
        dataset_text_field=training_args.dataset_text_field,
        processing_class=tokenizer,
        max_length=training_args.max_length,
        packing=training_args.packing,
        formatting_func=None,
        dataset_num_proc=training_args.dataset_num_proc,
        seed=training_args.seed,
        cache_dir=training_args.cache_dir,
    )

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
    )
    training_args.model_init_kwargs = model_kwargs

    ##################
    # Initialize model
    ##################
    logger.info("Initializing model")
    config = config_class(**model_config)
    model = causal_lm_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    ).to(getattr(torch, torch_dtype)) if model_args.model_name_or_path is not None and model_args.model_name_or_path.endswith("checkpoint") else causal_lm_class(config=config)

    model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model structure: {model}")
    logger.info(f"Model parameters: {model_num_params}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    else:
        logger.info("No checkpoint found, starting training from scratch.")

    ######################
    # Set warmup callbacks
    ######################
    # Handle warmup phases based on model type
    callbacks = []
    if is_doge2:
        # Doge2 multi warmup handling
        warmup_steps = getattr(training_args, 'warmup_steps', 0)
        warmup_phase_steps = model_config.get('warmup_phase_steps', None)  # Custom phase steps if specified
        if warmup_steps > 0:
            # Use the new multiphase warmup callback that handles all phases automatically
            callbacks.append(Doge2WarmupCallback(warmup_steps, warmup_phase_steps))
    else:
        # Doge MoE warmup handling
        if hasattr(config, 'is_moe') and config.is_moe and training_args.warmup_steps > 0:
            callbacks.append(DogeWarmupCallback(training_args.warmup_steps))

    ################################
    # Initialize the PT trainer
    ################################
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks if len(callbacks) > 0 else None,
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

    #############################
    # Register the model and save
    #############################
    register_model_classes(recipe_type)
    config_class.register_for_auto_class()
    model_class.register_for_auto_class("AutoModel")
    causal_lm_class.register_for_auto_class("AutoModelForCausalLM")
    tokenizer = AutoTokenizer.from_pretrained(f"{training_args.output_dir}")
    tokenizer.save_pretrained(f"{training_args.output_dir}")
    model = AutoModelForCausalLM.from_pretrained(f"{training_args.output_dir}")
    model.save_pretrained(f"{training_args.output_dir}")

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training finished! ***")


if __name__ == "__main__":
    model_config_parser = ArgumentParser()
    model_config_parser.add_argument(
        "--config", type=str, default="./recipes/doge/Doge-20M/config_full.yaml", help="path to yaml config file of PT"
    )

    parser = TrlParser((ScriptArguments, PTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    
    config_path = model_config_parser.parse_args().config
    model_config = yaml.load(
        open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )["model_config"]
    
    main(script_args, training_args, model_args, model_config)
