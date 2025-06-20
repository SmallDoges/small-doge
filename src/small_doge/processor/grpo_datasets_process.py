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

from typing import List, Dict, Optional, Union
import logging
import re
from datasets import Dataset, IterableDataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_dataset_ratios(datasets_and_ratios: List[Dict[str, float]]) -> None:
    """Validate that dataset ratios are properly formatted and sum to 1.0."""
    if not datasets_and_ratios:
        raise ValueError("datasets_and_ratios cannot be empty")
    
    total_ratio = 0.0
    for dataset_dict in datasets_and_ratios:
        if not isinstance(dataset_dict, dict) or len(dataset_dict) != 1:
            raise ValueError("Each item in datasets_and_ratios must be a dictionary with exactly one key-value pair")
        
        ratio = list(dataset_dict.values())[0]
        if not isinstance(ratio, (int, float)) or ratio <= 0:
            raise ValueError(f"Ratio must be a positive number, got {ratio}")
        
        total_ratio += ratio
    
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Total ratio must be 1.0, but got {total_ratio}. Please check your ratios.")


def load_grpo_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None
) -> Union[Dataset, DatasetDict]:
    """
    Load a GRPO dataset from HuggingFace Hub or local disk.
    
    Args:
        dataset_name: Name of the dataset (HF Hub format) or path to local dataset
        dataset_config: Configuration name for the dataset
        split: Dataset split to load
    
    Returns:
        Loaded dataset
    """
    try:
        if re.match(r'^[^/]+/[^/]+$', dataset_name):
            # HuggingFace Hub dataset
            logger.info(f"Loading dataset from HuggingFace Hub: {dataset_name}")
            if dataset_config:
                dataset = load_dataset(dataset_name, name=dataset_config, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
        else:
            # Local dataset
            logger.info(f"Loading dataset from local path: {dataset_name}")
            dataset = load_from_disk(dataset_name)
            if split and isinstance(dataset, DatasetDict):
                dataset = dataset[split]
                
        logger.info(f"Successfully loaded dataset: {dataset_name}")
        if isinstance(dataset, Dataset):
            logger.info(f"Dataset size: {len(dataset)}")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
        raise


def prepare_grpo_dataset(
    dataset: Union[Dataset, IterableDataset],
    dataset_name: str,
    system_prompt: Optional[str] = None,
) -> Union[Dataset, IterableDataset]:
    """
    Prepare a dataset for GRPO training.
    
    Expected format for GRPO datasets:
    - 'problem': The math problem or query
    - 'solution': The ground truth solution (optional, for reward calculation)
    
    Args:
        dataset: Input dataset
        dataset_name: Name of the dataset for logging
        system_prompt: Optional system prompt to prepend
    
    Returns:
        Processed dataset with 'prompt' field
    """
    logger.info(f"Preparing GRPO dataset: {dataset_name}")
    
    def preprocess_function(example):
        prompt = []
        if system_prompt is not None:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": example["problem"]})
        
        processed = {"prompt": prompt}
        
        # Keep solution if available for reward calculation
        if "solution" in example:
            processed["solution"] = example["solution"]
            
        return processed
    
    # Apply preprocessing
    if isinstance(dataset, Dataset):
        processed_dataset = dataset.map(
            preprocess_function,
            desc=f"Preprocessing {dataset_name} for GRPO"
        )
    else:
        processed_dataset = dataset.map(preprocess_function)
    
    # Remove unnecessary columns
    columns_to_remove = []
    for col in processed_dataset.column_names:
        if col not in ["prompt", "solution"]:
            columns_to_remove.append(col)
    
    if columns_to_remove:
        processed_dataset = processed_dataset.remove_columns(columns_to_remove)
    
    logger.info(f"Successfully prepared GRPO dataset: {dataset_name}")
    return processed_dataset


def mix_datasets_by_ratio(
    datasets_and_ratios: List[Dict[str, float]],
    total_sample_size: Optional[int] = None,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dataset:
    """
    Mix multiple datasets according to specified ratios for GRPO training.
    
    Args:
        datasets_and_ratios: List of dicts with dataset names and ratios
        total_sample_size: Total number of samples to use
        dataset_config: Configuration name for datasets
        split: Split to use from datasets
        system_prompt: Optional system prompt
    
    Returns:
        Mixed dataset ready for GRPO training
    """
    validate_dataset_ratios(datasets_and_ratios)
    
    logger.info("Starting GRPO dataset mixing process")
    logger.info(f"Datasets and ratios: {datasets_and_ratios}")
    logger.info(f"Total sample size: {total_sample_size}")
    
    mixed_datasets = []
    
    for dataset_dict in datasets_and_ratios:
        dataset_name = list(dataset_dict.keys())[0]
        ratio = dataset_dict[dataset_name]
        
        logger.info(f"Processing dataset: {dataset_name} with ratio: {ratio}")
        
        # Load dataset
        dataset = load_grpo_dataset(dataset_name, dataset_config, split)
        
        # Prepare dataset for GRPO
        dataset = prepare_grpo_dataset(dataset, dataset_name, system_prompt)
        
        # Calculate number of samples for this dataset
        if total_sample_size:
            num_samples = int(total_sample_size * ratio)
            if num_samples > len(dataset):
                logger.warning(
                    f"Requested {num_samples} samples from {dataset_name}, "
                    f"but dataset only has {len(dataset)} samples. Using all available samples."
                )
                num_samples = len(dataset)
            
            # Sample the dataset
            if num_samples < len(dataset):
                dataset = dataset.select(range(num_samples))
                logger.info(f"Sampled {num_samples} from {dataset_name}")
            else:
                logger.info(f"Using all {len(dataset)} samples from {dataset_name}")
        
        mixed_datasets.append(dataset)
        logger.info(f"Successfully processed {dataset_name}: {len(dataset)} samples")
    
    # Concatenate all datasets
    logger.info("Concatenating all datasets")
    final_dataset = concatenate_datasets(mixed_datasets)
    
    # Shuffle the mixed dataset
    final_dataset = final_dataset.shuffle(seed=42)
    
    logger.info(f"Successfully mixed GRPO datasets. Final dataset size: {len(final_dataset)}")
    logger.info(f"Dataset columns: {final_dataset.column_names}")
    
    return final_dataset
