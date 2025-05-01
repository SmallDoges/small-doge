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
from small_doge.models.doge2.modeling_doge2 import Doge2Config, Doge2ForCausalLM, Doge2Model
from trl import ModelConfig, ScriptArguments, TrlParser


logger = logging.getLogger(__name__)


def set_warmup_phase(model: Doge2ForCausalLM, phase: int):
    """
    Set the warm-up phase for model parameters
    
    Args:
        model: The model
        phase: Warm-up phase(1: Self-Attention, 2: MLP, 3: Residual, 4: All Parameters)
    
    Returns:
        model: The model with frozen/unfrozen parameters
    """

    # Define parameter patterns for each phase
    attn_params = [
        r"^model\.layers\.\d+\.self_attn\.A$",
        r"^model\.layers\.\d+\.self_attn\.dt_proj\.weight$",
        r"^model\.layers\.\d+\.self_attn\.q_norm\.weight$",
        r"^model\.layers\.\d+\.self_attn\.k_norm\.weight$",
    ]
    mlp_params = [
        r"^model\.layers\.\d+\.mlp\.router_gate\.weight$",
        r"^model\.layers\.\d+\.mlp\.down_embed\.weight$",
        r"^model\.layers\.\d+\.mlp\.up_embed\.weight$",
    ]
    residual_params = [
        r"^model\.layers\.\d+\.input_residual$",
        r"^model\.layers\.\d+\.post_attention_residual$",
    ]
    all_params = [r".*"]

    # Freeze all parameters first
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Thaw the corresponding parameters according to the current phase
    unfreeze_params = []
    phase_name = ""

    if phase == 1:
        # Phase 1: Only train self-attention-related parameters
        active_params = attn_params
        phase_name = "Self-Attention"
    elif phase == 2:
        # Phase 2: Only train MLP-related parameters
        active_params = mlp_params
        phase_name = "MLP"
    elif phase == 3:
        # Phase 3: Train residual connection related parameters
        active_params = residual_params
        phase_name = "Residual"
    elif phase == 4:
        # Phase 4: Train all parameters together
        active_params = all_params
        phase_name = "All Parameters"
    else:
        logger.warning(f"Invalid warm-up phase: {phase}, defaulting to all parameters")
        active_params = all_params
        phase_name = "All parameters"
    
    # Unfreeze the parameters of the current phase
    for name, param in model.named_parameters():
        if any(re.match(pattern, name) for pattern in active_params):
            param.requires_grad = True
            unfreeze_params.append(name)
    
    logger.info(f"Warm-up phase {phase} ({phase_name}): unfreeze {len(unfreeze_params)} parameters, freeze other parameters")
    logger.info(f"Unfrozen parameters: {unfreeze_params}")
    return model


class MultiphaseWarmupCallback(TrainerCallback):
    """
    Multiphase warm-up callback with automatic phase transition
    """

    def __init__(self, warmup_phase_steps: list):
        """
        Args:
            warmup_phase_steps: Steps for each warm-up phase [phase1_steps, phase2_steps, phase3_steps, phase4_steps]
        """
        self.warmup_phase_steps = warmup_phase_steps
        self.total_steps = sum(warmup_phase_steps)
        self.phase_end_steps = [sum(warmup_phase_steps[:i+1]) for i in range(len(warmup_phase_steps))]
        self.current_phase = 1

        logger.info(f"Multiphase warm-up: {len(warmup_phase_steps)} phases, steps per phase: {warmup_phase_steps}")
        logger.info(f"Phase transition points: {self.phase_end_steps}")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: Doge2ForCausalLM = None, **kwargs):
        if model is not None:
            set_warmup_phase(model, self.current_phase)
            logger.info(f"Starting warm-up phase 1: Self-Attention")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: Doge2ForCausalLM = None, **kwargs):
        if model is None:
            return

        current_step = state.global_step
        determined_phase = 1 # Default to phase 1

        for i, end_step in enumerate(self.phase_end_steps):
            if current_step < end_step:
                determined_phase = i + 1
                break
            elif i == len(self.phase_end_steps) - 1:
                determined_phase = len(self.phase_end_steps) # Last phase

        # If the phase has changed, update model parameters
        if determined_phase != self.current_phase:
            phase_names = ["Self-Attention", "MLP", "Residual", "All Parameters"]
            logger.info(f"Transitioning from warm-up phase {self.current_phase}: {phase_names[self.current_phase-1]} "
                       f"to phase {determined_phase}: {phase_names[determined_phase-1]} at step {current_step}")
            
            self.current_phase = determined_phase
            set_warmup_phase(model, self.current_phase)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: Doge2ForCausalLM = None, **kwargs):
        # Check if the current step is the last step of the warm-up phase
        if state.global_step == self.total_steps:
            logger.info("All warm-up phases completed, transitioning to normal training mode")
            # Unfreeze all parameters for normal training
            if model is not None:
                for name, param in model.named_parameters():
                    param.requires_grad = True
                logger.info("All parameters unfrozen for normal training")


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
    config = Doge2Config(**model_config)
    model = Doge2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    ) if model_args.model_name_or_path is not None and model_args.model_name_or_path.endswith("checkpoint") else Doge2ForCausalLM(config=config)

    model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model structure: {model}")
    logger.info(f"Model parameters: {model_num_params}")

    #########################
    # Multiphase warmup phase
    #########################
    if training_args.warmup_steps > 0:
        total_warmup_steps = training_args.warmup_steps
        warmup_phase_steps = [total_warmup_steps // 4] * 4
        warmup_phase_steps[-1] += total_warmup_steps % 4
        warmup_callback = MultiphaseWarmupCallback(warmup_phase_steps)
        logger.info(f"Initialized automatic multiphase warm-up with 4 phases")
        logger.info(f"Phase steps: {warmup_phase_steps}, total warm-up steps: {total_warmup_steps}")
    else:
        warmup_callback = None

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
        callbacks=[warmup_callback] if training_args.warmup_steps > 0 else None,
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
    AutoConfig.register("doge2", Doge2Config)
    AutoModel.register(Doge2Config, Doge2Model)
    AutoModelForCausalLM.register(Doge2Config, Doge2ForCausalLM)
    Doge2Config.register_for_auto_class()
    Doge2Model.register_for_auto_class("AutoModel")
    Doge2ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
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
        "--config", type=str, default="./recipes/doge2/Doge-160M/config_full.yaml", help="path to yaml config file of PT"
    )

    parser = TrlParser((ScriptArguments, TrainingArguments, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_config = yaml.load(
        open(model_config_parser.parse_args().config, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )["model_config"]
    main(script_args, training_args, model_args, model_config)
