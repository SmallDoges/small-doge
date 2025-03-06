# Copyright 2025 The SamllDoge Team. All rights reserved.
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
import re
import os
import sys
from argparse import ArgumentParser

import datasets
import torch
import transformers
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerControl,
    TrainerState,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import yaml
from small_doge.models.modeling_doge import DogeConfig, DogeForCausalLM, DogeModel
from trl import ModelConfig, ScriptArguments, TrlParser


logger = logging.getLogger(__name__)


def set_moe_warmup_phase(model: DogeForCausalLM):
    MoE_params = [
        r"^model\.layers\.\d+\.feed_forward\.queries_proj\.weight$",
        r"^model\.layers\.\d+\.feed_forward\.keys$",
        r"^model\.layers\.\d+\.feed_forward\.down_embed\.weight$",
        r"^model\.layers\.\d+\.feed_forward\.up_embed\.weight$",
        r"^model\.layers\.\d+\.feed_forward\.mlp_scaling$",
        r"^model\.layers\.\d+\.feed_forward\.moe_scaling$",
    ]

    # Freeze all parameters first
    unfreeze_params = []
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Then unfreeze the target MoE parameters
    for name, param in model.named_parameters():
        if any(re.match(pattern, name) for pattern in MoE_params):
            param.requires_grad = True
            unfreeze_params.append(name)

    logger.info(f"MoE warm-up phase: unfreeze {unfreeze_params}, freeze other parameters")
    return model

class MoEWarmupCallback(TrainerCallback):
    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
        logger.info(f"MoE warm-up phase, only train specific parameters, until step {warmup_steps}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == self.warmup_steps:
            control.should_training_stop = True
            logger.info("MoE warm-up phase finished, please set warmup_steps to 0 in config, and restart training")


def main(script_args, training_args, model_args, model_config):
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

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    dataset = load_from_disk(script_args.dataset_name)

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
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

    ################################
    # Initialize model
    ################################
    logger.info("Initializing model")
    config = DogeConfig(**model_config)
    model = DogeForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    ) if model_args.model_name_or_path is not None and model_args.model_name_or_path.endswith("checkpoint") else DogeForCausalLM(config=config)

    model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model structure: {model}")
    logger.info(f"Model parameters: {model_num_params}")

    if config.is_moe and training_args.warmup_steps > 0:
        model = set_moe_warmup_phase(model)

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
        callbacks=[MoEWarmupCallback(training_args.warmup_steps)] if config.is_moe and training_args.warmup_steps > 0 else None,
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

    ################################
    # Register the model and save
    ################################
    AutoConfig.register("doge", DogeConfig)
    AutoModel.register(DogeConfig, DogeModel)
    AutoModelForCausalLM.register(DogeConfig, DogeForCausalLM)
    DogeConfig.register_for_auto_class()
    DogeModel.register_for_auto_class("AutoModel")
    DogeForCausalLM.register_for_auto_class("AutoModelForCausalLM")
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

    parser = TrlParser((ScriptArguments, TrainingArguments, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_config = yaml.load(
        open(model_config_parser.parse_args().config, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )["model_config"]
    main(script_args, training_args, model_args, model_config)
