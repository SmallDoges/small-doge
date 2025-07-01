# coding=utf-8
# Copyright 2025 SmallDoge team. All rights reserved.
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
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


class DogeWarmupCallback(TrainerCallback):
    """
    MoE warmup callback for Doge model with automatic parameter management
    """
    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
        logger.info(f"MoE warm-up phase, only train specific parameters, until step {warmup_steps}")

    def _set_warmup_phase(self, model):
        """
        Set parameter freezing for Doge model MoE warmup phase
        
        Args:
            model: Doge model instance
            
        Returns:
            model: Model with parameter freezing settings applied
        """
        MoE_params = [
            r"^model\.layers\.\d+\.feed_forward\.router_gate\.weight$",
            r"^model\.layers\.\d+\.feed_forward\.down_embed\.weight$",
            r"^model\.layers\.\d+\.feed_forward\.up_embed\.weight$",
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

        logger.info(f"MoE warm-up phase: unfreeze {len(unfreeze_params)} parameters, freeze other parameters")
        return model

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Initialize MoE warmup phase at training start"""
        if model is not None:
            self._set_warmup_phase(model)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step == self.warmup_steps:
            control.should_training_stop = True
            logger.info("MoE warm-up phase finished, please set warmup_steps to 0 in config, and restart training")


class Doge2WarmupCallback(TrainerCallback):
    """
    Multiphase progressive warmup callback for Doge2 model with automatic phase transition
    """
    def __init__(self, warmup_steps: int, warmup_phase_steps: list = None):
        """
        Args:
            warmup_steps: Total warmup steps (for backward compatibility)
            warmup_phase_steps: Steps for each warm-up phase [phase1_steps, phase2_steps, phase3_steps, phase4_steps]
                              If None, warmup_steps will be divided equally among 4 phases
        """
        if warmup_phase_steps is None:
            # Divide warmup_steps equally among 4 phases for backward compatibility
            phase_steps = warmup_steps // 4
            self.warmup_phase_steps = [phase_steps] * 4
            self.warmup_phase_steps[-1] += warmup_steps % 4  # Add remainder to last phase
        else:
            self.warmup_phase_steps = warmup_phase_steps
            
        self.total_steps = sum(self.warmup_phase_steps)
        self.phase_end_steps = [sum(self.warmup_phase_steps[:i+1]) for i in range(len(self.warmup_phase_steps))]
        self.current_phase = 1
        self.phase_names = ["Self-Attention", "Self-Attention + MLP", "Self-Attention + MLP + Residual", "All Parameters"]

        logger.info(f"Doge2 multiphase warm-up: {len(self.warmup_phase_steps)} phases, steps per phase: {self.warmup_phase_steps}")
        logger.info(f"Phase transition points: {self.phase_end_steps}")
        logger.info(f"Total warmup steps: {self.total_steps}")

    def _set_warmup_phase(self, model, phase: int):
        """
        Set the warm-up phase for model parameters
        
        Args:
            model: The Doge2 model
            phase: Warm-up phase (1: Self-Attention, 2: Self-Attention+MLP, 3: Self-Attention+MLP+Residual, 4: All Parameters)
        
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

        # Freeze all parameters first
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Thaw the corresponding parameters according to the current phase
        unfreeze_params = []
        active_params = []

        if phase == 1:
            # Phase 1: Only train self-attention-related parameters
            active_params = attn_params
        elif phase == 2:
            # Phase 2: Train self-attention + MLP parameters
            active_params = attn_params + mlp_params
        elif phase == 3:
            # Phase 3: Train self-attention + MLP + residual parameters
            active_params = attn_params + mlp_params + residual_params
        elif phase == 4:
            # Phase 4: Train all parameters together
            active_params = [r".*"]
        else:
            logger.warning(f"Invalid warm-up phase: {phase}, defaulting to all parameters")
            active_params = [r".*"]

        # Unfreeze the corresponding parameters
        for name, param in model.named_parameters():
            if any(re.match(pattern, name) for pattern in active_params):
                param.requires_grad = True
                unfreeze_params.append(name)

        logger.info(f"Doge2 warm-up phase {phase} ({self.phase_names[phase-1]}): unfreeze {len(unfreeze_params)} parameters")
        return model

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Initialize the first warmup phase at training start"""
        if model is not None:
            self._set_warmup_phase(model, self.current_phase)
            logger.info(f"Starting Doge2 warm-up phase 1: {self.phase_names[0]}")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Check and handle phase transitions at the beginning of each step"""
        if model is None:
            return

        current_step = state.global_step
        determined_phase = 1  # Default to phase 1

        # Determine which phase we should be in based on current step
        for i, end_step in enumerate(self.phase_end_steps):
            if current_step < end_step:
                determined_phase = i + 1
                break
            elif i == len(self.phase_end_steps) - 1:
                determined_phase = len(self.phase_end_steps)  # Last phase

        # If the phase has changed, update model parameters
        if determined_phase != self.current_phase:
            logger.info(f"Transitioning from Doge2 warm-up phase {self.current_phase}: {self.phase_names[self.current_phase-1]} "
                       f"to phase {determined_phase}: {self.phase_names[determined_phase-1]} at step {current_step}")
            
            self.current_phase = determined_phase
            self._set_warmup_phase(model, self.current_phase)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Check if warmup is complete and transition to normal training"""
        # Check if the current step is the last step of the warm-up phase
        if state.global_step == self.total_steps:
            logger.info("All Doge2 warm-up phases completed, transitioning to normal training mode")
            # Unfreeze all parameters for normal training
            if model is not None:
                for name, param in model.named_parameters():
                    param.requires_grad = True
                logger.info("All parameters unfrozen for normal training")
