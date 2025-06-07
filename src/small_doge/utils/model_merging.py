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

import os
import torch
import logging
from typing import List, Dict, Optional, Union, Tuple
from enum import Enum
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerControl, TrainerState


logger = logging.getLogger(__name__)


class MergeMethod(Enum):
    """Enumeration of supported model merging methods."""
    SMA = "sma"  # Simple Moving Average
    EMA = "ema"  # Exponential Moving Average
    WMA = "wma"  # Weighted Moving Average


def load_checkpoint(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a model checkpoint from the given path.
    
    Args:
        ckpt_path (str): Path to the checkpoint file.
        
    Returns:
        Dict[str, torch.Tensor]: Model state dictionary.
    """
    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    return state_dict


def simple_moving_average(
    checkpoints: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Perform Simple Moving Average (SMA) model merging.
    All models receive equal weights.
    
    Args:
        checkpoints (List[Dict[str, torch.Tensor]]): List of model state dictionaries.
        
    Returns:
        Dict[str, torch.Tensor]: Merged model state dictionary.
    """
    if not checkpoints:
        raise ValueError("No checkpoints provided for merging.")
    
    merged_state = {}
    n_models = len(checkpoints)
    
    # Get the keys from the first checkpoint
    keys = checkpoints[0].keys()
    
    # Sum all parameters
    for key in keys:
        merged_state[key] = sum(ckpt[key] for ckpt in checkpoints) / n_models
        
    return merged_state


def exponential_moving_average(
    checkpoints: List[Dict[str, torch.Tensor]], 
    alpha: float = 0.2
) -> Dict[str, torch.Tensor]:
    """
    Perform Exponential Moving Average (EMA) model merging.
    Uses the formula w_i = α(1-α)^(N-i) as described in the paper.
    
    Args:
        checkpoints (List[Dict[str, torch.Tensor]]): List of model state dictionaries,
                                                   ordered from oldest to newest.
        alpha (float): Smoothing factor between 0 and 1.
        
    Returns:
        Dict[str, torch.Tensor]: Merged model state dictionary.
    """
    if not checkpoints:
        raise ValueError("No checkpoints provided for merging")
    
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    n_models = len(checkpoints)
    merged_state = {}
    keys = checkpoints[0].keys()
    
    # Calculate weights according to w_i = α(1-α)^(N-i)
    weights = [alpha * ((1-alpha) ** (n_models - i - 1)) for i in range(n_models)]
    
    # Normalize weights to ensure they sum to 1
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]
    
    # Log the weights for debugging
    logger.info(f"EMA weights: {normalized_weights}")
    
    # Apply weighted average
    for key in keys:
        merged_state[key] = sum(w * ckpt[key] for w, ckpt in zip(normalized_weights, checkpoints))
    
    return merged_state


def weighted_moving_average(
    checkpoints: List[Dict[str, torch.Tensor]], 
    weights: Optional[List[float]] = None
) -> Dict[str, torch.Tensor]:
    """
    Perform Weighted Moving Average (WMA) model merging.
    
    Args:
        checkpoints (List[Dict[str, torch.Tensor]]): List of model state dictionaries.
        weights (Optional[List[float]]): List of weights for each checkpoint.
                                       If None, uses linear weights (i+1).
        
    Returns:
        Dict[str, torch.Tensor]: Merged model state dictionary.
    """
    if not checkpoints:
        raise ValueError("No checkpoints provided for merging.")
    
    n_models = len(checkpoints)
    
    # Default to linear weights if none provided
    if weights is None:
        weights = [i + 1 for i in range(n_models)]  # Linear weights [1, 2, 3, ...]
    
    if len(weights) != n_models:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of checkpoints ({n_models}).")
    
    # Normalize weights
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]
    
    logger.info(f"WMA weights: {normalized_weights}")
    
    merged_state = {}
    keys = checkpoints[0].keys()
    
    # Weighted sum of parameters
    for key in keys:
        merged_state[key] = sum(w * ckpt[key] for w, ckpt in zip(normalized_weights, checkpoints))
    
    return merged_state


class ModelMerger:
    """
    A class to handle model merging operations with different strategies.
    
    This class provides a unified interface to merge model checkpoints or 
    state dictionaries using various averaging methods as described in the
    PMA (Pre-trained Model Average) paper https://arxiv.org/pdf/2505.12082.
    """
    
    def __init__(self, method: Union[str, MergeMethod] = MergeMethod.SMA, 
                 alpha: float = 0.2, 
                 custom_weights: Optional[List[float]] = None):
        """
        Initialize the ModelMerger.
        
        Args:
            method: Merging method, one of "sma", "ema", "wma" or MergeMethod enum
            alpha: Smoothing factor for EMA (default: 0.2 as recommended in the paper)
            custom_weights: Custom weights for WMA (if None, linear weights will be used)
        """
        if isinstance(method, str):
            try:
                self.method = MergeMethod(method.lower())
            except ValueError:
                raise ValueError(f"Unknown merging method: {method}. Use 'sma', 'ema', or 'wma'.")
        else:
            self.method = method
            
        self.alpha = alpha
        self.custom_weights = custom_weights
        
    def merge_checkpoints(self, checkpoint_paths: List[str], 
                          save_path: Optional[str] = None,
                          interval: Optional[int] = None,
                          num_checkpoints: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Merge model checkpoints from filesystem paths.
        
        Args:
            checkpoint_paths: List of paths to checkpoint files
            save_path: Path to save the merged checkpoint (optional)
            interval: Token interval between checkpoints to select (if None, use all)
            num_checkpoints: Number of checkpoints to use (if None, use all)
            
        Returns:
            Dict[str, torch.Tensor]: Merged model state dictionary
        """
        # Filter checkpoints based on interval if specified
        if interval is not None and len(checkpoint_paths) > interval:
            filtered_paths = [checkpoint_paths[i] for i in range(0, len(checkpoint_paths), interval)]
            checkpoint_paths = filtered_paths
            logger.info(f"Selected {len(checkpoint_paths)} checkpoints with interval {interval}")
            
        # Select most recent checkpoints if num_checkpoints is specified
        if num_checkpoints is not None and num_checkpoints < len(checkpoint_paths):
            checkpoint_paths = checkpoint_paths[-num_checkpoints:]
            logger.info(f"Using {num_checkpoints} most recent checkpoints")
            
        # Load all checkpoints
        checkpoints = []
        for path in checkpoint_paths:
            try:
                checkpoints.append(load_checkpoint(path))
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {path}: {e}")
                continue
                
        if not checkpoints:
            raise ValueError("No valid checkpoints could be loaded")
        
        return self._merge_state_dicts(checkpoints, save_path)
    
    def merge_state_dicts(self, state_dicts: List[Dict[str, torch.Tensor]], 
                         save_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Merge model state dictionaries directly.
        
        Args:
            state_dicts: List of model state dictionaries
            save_path: Path to save the merged state dict (optional)
            
        Returns:
            Dict[str, torch.Tensor]: Merged model state dictionary
        """
        return self._merge_state_dicts(state_dicts, save_path)
    
    def merge_models(self, models: List[torch.nn.Module], 
                    save_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Merge PyTorch models directly.
        
        Args:
            models: List of PyTorch models
            save_path: Path to save the merged state dict (optional)
            
        Returns:
            Dict[str, torch.Tensor]: Merged model state dictionary
        """
        state_dicts = [model.state_dict() for model in models]
        return self._merge_state_dicts(state_dicts, save_path)
    
    def _merge_state_dicts(self, state_dicts: List[Dict[str, torch.Tensor]], 
                          save_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Internal method to merge state dictionaries based on the selected method.
        
        Args:
            state_dicts: List of model state dictionaries
            save_path: Path to save the merged state dict (optional)
            
        Returns:
            Dict[str, torch.Tensor]: Merged model state dictionary
        """
        if not state_dicts:
            raise ValueError("No state dictionaries provided for merging.")
        
        # Verify parameter structure consistency
        keys = set(state_dicts[0].keys())
        for i, sd in enumerate(state_dicts[1:], 1):
            if set(sd.keys()) != keys:
                logger.warning(f"Checkpoint {i} has different parameter structure")
                # Find missing and extra keys
                missing = keys - set(sd.keys())
                extra = set(sd.keys()) - keys
                if missing:
                    logger.warning(f"Missing keys in checkpoint {i}: {missing}")
                if extra:
                    logger.warning(f"Extra keys in checkpoint {i}: {extra}")
        
        # Calculate weights for WMA if not provided
        weights = None
        if self.method == MergeMethod.WMA:
            weights = self.custom_weights if self.custom_weights else [i + 1 for i in range(len(state_dicts))]
        
        # Select merging method
        if self.method == MergeMethod.SMA:
            merged_state = simple_moving_average(state_dicts)
        elif self.method == MergeMethod.EMA:
            merged_state = exponential_moving_average(state_dicts, self.alpha)
        elif self.method == MergeMethod.WMA:
            merged_state = weighted_moving_average(state_dicts, weights)
        
        # Save merged checkpoint if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(merged_state, save_path)
            logger.info(f"Saved merged model to {save_path}")
        
        return merged_state
    
    @staticmethod
    def pma_init(checkpoint_paths: List[str],
                num_models: int = 10,
                method: str = "sma",
                alpha: float = 0.2,
                save_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Implementation of Pre-trained Model Average (PMA) initialization.
        
        This is a convenience static method that creates a ModelMerger instance
        and performs merging in one step.
        
        Args:
            checkpoint_paths: Paths to checkpoint files
            num_models: Number of most recent models to merge
            method: Merging method, one of "sma", "ema", or "wma"
            alpha: Smoothing factor for EMA
            save_path: Path to save the merged checkpoint
            
        Returns:
            Dict[str, torch.Tensor]: Merged model state dictionary
        """
        merger = ModelMerger(method=method, alpha=alpha)
        return merger.merge_checkpoints(
            checkpoint_paths=checkpoint_paths,
            num_checkpoints=num_models,
            save_path=save_path
        )


class PMACallback(TrainerCallback):
    """
    Callback for Hugging Face Trainer that automatically performs PMA merging.
    
    This callback can be used to:
    1. Periodically save merged checkpoints during training
    2. Recover from training instabilities using PMA-init
    """
    
    def __init__(self, 
                output_dir: str,
                merge_interval: int = 10000,
                num_checkpoints: int = 10,
                method: str = "sma",
                alpha: float = 0.2,
                stability_monitor: bool = False,
                loss_spike_threshold: float = 1.5):
        """
        Initialize the PMA callback.
        
        Args:
            output_dir: Directory to save merged checkpoints
            merge_interval: Steps between model merging operations
            num_checkpoints: Number of checkpoints to merge
            method: Merging method, one of "sma", "ema", or "wma"
            alpha: Smoothing factor for EMA
            stability_monitor: Whether to monitor training stability
            loss_spike_threshold: Threshold for detecting loss spikes
        """
        self.output_dir = output_dir
        self.merge_interval = merge_interval
        self.num_checkpoints = num_checkpoints
        self.method = method
        self.alpha = alpha
        self.stability_monitor = stability_monitor
        self.loss_spike_threshold = loss_spike_threshold
        self.checkpoint_paths = []
        self.loss_history = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"PMA Callback initialized with {method} method, merging every {merge_interval} steps")
        
    def on_save(self, args: TrainingArguments, state: TrainerState, 
               control: TrainerControl, **kwargs):
        """Save checkpoint path when model is saved."""
        if hasattr(state, "best_model_checkpoint"):
            checkpoint_path = state.best_model_checkpoint
            if checkpoint_path and checkpoint_path not in self.checkpoint_paths:
                self.checkpoint_paths.append(checkpoint_path)
                logger.info(f"Added checkpoint to PMA tracking: {checkpoint_path}")
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, 
                   control: TrainerControl, **kwargs):
        """Perform periodic model merging or stability recovery."""
        # Track loss for stability monitoring
        if self.stability_monitor and hasattr(state, "log_history") and state.log_history:
            if "loss" in state.log_history[-1]:
                self.loss_history.append(state.log_history[-1]["loss"])
                
                # Check for loss spike
                if len(self.loss_history) >= 2:
                    if self.loss_history[-1] > self.loss_history[-2] * self.loss_spike_threshold:
                        logger.warning(f"Loss spike detected: {self.loss_history[-2]} -> {self.loss_history[-1]}")
                        if len(self.checkpoint_paths) >= self.num_checkpoints:
                            # Perform PMA-init for recovery
                            self._perform_pma_recovery(kwargs.get("model", None), kwargs.get("optimizer", None))
        
        # Periodic merging
        if state.global_step > 0 and state.global_step % self.merge_interval == 0:
            if len(self.checkpoint_paths) >= self.num_checkpoints:
                self._perform_model_merging(state.global_step)
            else:
                logger.info(f"Not enough checkpoints for merging ({len(self.checkpoint_paths)}/{self.num_checkpoints})")
    
    def _perform_model_merging(self, step: int):
        """Perform model merging and save the result."""
        logger.info(f"Performing model merging at step {step}")
        merger = ModelMerger(method=self.method, alpha=self.alpha)
        
        save_path = os.path.join(self.output_dir, f"merged_model_step_{step}.pt")
        merger.merge_checkpoints(
            checkpoint_paths=self.checkpoint_paths[-self.num_checkpoints:],
            save_path=save_path
        )
        
        logger.info(f"Model merging complete at step {step}, saved to {save_path}")
        
    def _perform_pma_recovery(self, model, optimizer):
        """Recover from training instability using PMA-init."""
        if model is None:
            logger.warning("Cannot perform PMA recovery: model not provided")
            return
            
        logger.info("Performing PMA-init for training recovery")
        merger = ModelMerger(method=self.method, alpha=self.alpha)
        merged_state = merger.merge_checkpoints(
            checkpoint_paths=self.checkpoint_paths[-self.num_checkpoints:],
        )
        
        # Apply merged weights to model
        model.load_state_dict(merged_state)
        
        # Reset optimizer state if provided
        if optimizer is not None:
            optimizer.state = {}
            
        logger.info("PMA recovery complete")


# Command-line interface for model merging
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Merging Tool")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths to checkpoint files")
    parser.add_argument("--method", choices=["sma", "ema", "wma"], default="sma", help="Merging method")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for EMA")
    parser.add_argument("--num-checkpoints", type=int, help="Number of checkpoints to use")
    parser.add_argument("--interval", type=int, help="Interval between checkpoints to use")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    
    # Perform merging
    ModelMerger.pma_init(
        checkpoint_paths=args.checkpoints,
        num_models=args.num_checkpoints if args.num_checkpoints else len(args.checkpoints),
        method=args.method,
        alpha=args.alpha,
        save_path=args.output
    )