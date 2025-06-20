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

from typing import List, Dict, Optional, Union, Callable
import json
import logging
import warnings
import re
from datasets import Dataset, IterableDataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl.data_utils import maybe_apply_chat_template
from argparse import ArgumentParser


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


def validate_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    """Validate tokenizer configuration."""
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        logger.warning("Tokenizer has no pad_token, using eos_token as pad_token")
        tokenizer.pad_token = tokenizer.eos_token


def prepare_dataset(
    dataset: Union[Dataset, IterableDataset],
    dataset_name: str,
    processing_class: Union[PreTrainedTokenizerBase],
    formatting_func: Optional[Callable[[dict], str]],
    dataset_num_proc: Optional[int],
    tools: Optional[List[dict]] = None,
) -> Union[Dataset, IterableDataset]:
    """
    Prepare GRPO dataset for training.
    
    Args:
        dataset: The dataset to prepare.
        dataset_name: Name of the dataset for logging.
        processing_class: Tokenizer class used for processing.
        formatting_func: Optional formatting function.
        dataset_num_proc: Number of processes for dataset processing.
        tools: Optional tools for chat template.
    
    Returns:
        Prepared dataset with prompt field for GRPO training.
    """
    # Build the kwargs for the `map` function
    map_kwargs = {"writer_batch_size": 10}
    if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
        map_kwargs["num_proc"] = dataset_num_proc
    
    # Check if dataset has required columns for GRPO
    first_example = next(iter(dataset))
    required_columns = ["problem"]  # GRPO requires problem field
    missing_columns = [col for col in required_columns if col not in first_example.keys()]
    
    if missing_columns:
        raise ValueError(
            f"GRPO dataset must contain columns: {required_columns}. "
            f"Missing columns: {missing_columns}. "
            f"Available columns: {list(first_example.keys())}"
        )
    
    # Apply the formatting function if any
    if formatting_func is not None:
        if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

        batched = isinstance(formatting_func(next(iter(dataset))), list)

        def _func(example):
            formatted = formatting_func(example)
            if isinstance(formatted, dict):
                return formatted
            else:
                # If formatting function returns a string, assume it's for prompt
                return {"prompt": formatted, "problem": example.get("problem", ""), "solution": example.get("solution", "")}

        dataset = dataset.map(_func, batched=batched, **map_kwargs)

    # Transform problem to prompt format for GRPO
    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
        map_kwargs["desc"] = f"Converting problem to prompt in {dataset_name} dataset"
    
    def convert_to_prompt(example):
        """Convert problem field to prompt format for GRPO."""
        prompt = [{"role": "user", "content": example["problem"]}]
        result = {"prompt": prompt}
        
        # Keep solution if available for reward calculation
        if "solution" in example:
            result["solution"] = example["solution"]
            
        return result
    
    dataset = dataset.map(convert_to_prompt, **map_kwargs)

    # Apply the chat template if needed
    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
        map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
    dataset = dataset.map(
        maybe_apply_chat_template, 
        fn_kwargs={"tokenizer": processing_class, "tools": tools}, 
        **map_kwargs
    )

    return dataset


def mix_datasets_by_ratio(
    datasets_and_ratios: List[Dict[str, float]],
    total_sample_size: int,
    processing_class: Union[PreTrainedTokenizerBase],
    formatting_func: Optional[Callable[[dict], str]],
    dataset_num_proc: Optional[int],
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
    tools: Optional[List[dict]] = None,
):
    """
    Mix multiple GRPO datasets by ratio.

    Args:
        datasets_and_ratios: List of dictionaries, each containing a dataset and its ratio.
            Each dictionary contains one key-value pair where key is the dataset name and value is the mixing ratio.
        total_sample_size: Total sample size for the mixed training dataset.
        processing_class: Tokenizer class used for processing the text.
        formatting_func: Optional formatting function to convert dataset entries to the desired format.
        dataset_num_proc: Number of processes to use for dataset processing.
        seed: Random seed for dataset shuffling to ensure reproducibility.
        cache_dir: Directory to cache the datasets.
        tools: Optional tools for chat template.
    
    Returns:
        DatasetDict: A dictionary containing all mixed and processed dataset splits.
    
    Example:
    ```python
        from transformers import AutoTokenizer

        # Define datasets and their mixing ratios
        datasets_and_ratios = [
            {"lighteval/MATH": 1.0},
        ]

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")
    
        # Mix datasets
        mixed_dataset = mix_datasets_by_ratio(
            datasets_and_ratios=datasets_and_ratios,
            total_sample_size=10000,
            processing_class=tokenizer,
            formatting_func=None,
            dataset_num_proc=4,
            seed=42,
            cache_dir="./cache",
        )
        print(mixed_dataset)
    ```"""
    # Validate input parameters
    validate_dataset_ratios(datasets_and_ratios)
    
    # Check if the dataset ratios sum to 1.0 (redundant but kept for backwards compatibility)
    total_ratio = sum([list(dataset.values())[0] for dataset in datasets_and_ratios])
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Total ratio must be 1.0, but got {total_ratio}. Please check your ratios.")

    final_mixed_dataset = {}

    for dataset_and_ratio in datasets_and_ratios:
        dataset_name, ratio = dataset_and_ratio.popitem()

        # Check subset name
        windows_drive_pattern = r'^[a-zA-Z]:.*'
        is_windows_path = bool(re.match(windows_drive_pattern, dataset_name))
        if ":" in dataset_name and not is_windows_path:
            dataset_name, subset_name = dataset_name.split(":")
        else:
            subset_name = None

        # If the dataset is a string, load it from the hub or disk
        if isinstance(dataset_name, str):
            if re.match(r"^[^/]+/[^/]+$", dataset_name):
                dataset = load_dataset(dataset_name, name=subset_name, cache_dir=cache_dir)
            else:
                if subset_name is not None:
                    warnings.warn(
                        f"You passed a local dataset path, subsetting is not supported, ignoring subset name: {subset_name}",
                        UserWarning,
                    )
                dataset = load_from_disk(dataset_name)

        # Process each split of the dataset
        for split_name, split_dataset in dataset.items():

            logger.info(f"Original dataset size for {dataset_name}: {split_name}: {len(split_dataset)}")
            
            # Process the dataset for GRPO training
            split_dataset = prepare_dataset(
                split_dataset,
                dataset_name=f"{dataset_name}: {split_name}" if subset_name is None else f"{dataset_name}: {subset_name}: {split_name}",
                processing_class=processing_class,
                formatting_func=formatting_func,
                dataset_num_proc=dataset_num_proc,
                tools=tools,
            )

            # Calculate the target size for the dataset
            if total_sample_size == -1:
                target_size = len(split_dataset)
            else:
                target_size = int(total_sample_size * ratio) if split_name == "train" else len(split_dataset)
            current_size = len(split_dataset)
            logger.info(f"Processed dataset size for {dataset_name}: {split_name}: {current_size}")
            logger.info(f"Target size for {dataset_name}: {split_name}: {target_size}")

            # If the dataset is smaller than the target size, repeat it
            if current_size < target_size:
                logger.warning(
                    f"Dataset {dataset_name}: {split_name} is smaller than the target size. "
                    f"Repeating the dataset to reach the target size."
                )
                indices = []
                full_copies = target_size // current_size
                remainder = target_size % current_size

                for _ in range(full_copies):
                    indices.extend(range(current_size))
                if remainder > 0:
                    indices.extend(range(remainder))
                
                split_dataset = split_dataset.select(indices)
            else:
                logger.warning(
                    f"Dataset {dataset_name}: {split_name} is larger than the target size. "
                    f"Truncating the dataset to reach the target size."
                )
                split_dataset = split_dataset.select(range(target_size))
            
            # Concatenate the split dataset with the final mixed dataset
            if split_name in final_mixed_dataset:
                final_mixed_dataset[split_name] = concatenate_datasets(
                    [final_mixed_dataset[split_name], split_dataset]
                )
            else:
                final_mixed_dataset[split_name] = split_dataset
         
    # Shuffle the train dataset
    if "train" in final_mixed_dataset:
        final_mixed_dataset["train"] = final_mixed_dataset["train"].shuffle(seed)

    # Create a DatasetDict with the merged datasets
    final_dataset = DatasetDict(final_mixed_dataset)
    return final_dataset
            

def main(args):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    validate_tokenizer(tokenizer)

    # Mix datasets
    mixed_dataset = mix_datasets_by_ratio(
        datasets_and_ratios=args.datasets_and_ratios,
        total_sample_size=args.total_sample_size,
        processing_class=tokenizer,
        formatting_func=None,
        dataset_num_proc=args.dataset_num_proc,
        seed=args.seed,
        cache_dir=args.cache_dir,
        tools=args.tools,
    )
    
    # Save the mixed dataset
    mixed_dataset.save_to_disk(args.dataset_save_path)
    print(f"Mixed GRPO dataset saved to {args.dataset_save_path}")

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_and_ratios", type=str, required=True,
                          help="JSON string of list of dictionaries with dataset names and mixing ratios")
    argparser.add_argument("--dataset_save_path", type=str, required=True,
                          help="Path to save the mixed dataset")
    argparser.add_argument("--total_sample_size", type=int, required=True,
                          help="Total sample size for the mixed training dataset")
    argparser.add_argument("--tokenizer_name_or_path", type=str, required=True,
                          help="Tokenizer name or path")
    argparser.add_argument("--dataset_num_proc", type=int, default=4,
                          help="Number of processes for dataset processing")
    argparser.add_argument("--seed", type=int, default=42,
                          help="Random seed for reproducibility")
    argparser.add_argument("--cache_dir", type=str, default="./cache",
                          help="Directory to cache datasets")
    argparser.add_argument("--tools", type=str, default=None,
                          help="Tools for chat template (JSON string)")
    args = argparser.parse_args()

    # Parse datasets_and_ratios from JSON string
    args.datasets_and_ratios = json.loads(args.datasets_and_ratios)
    
    # Parse tools if provided
    if args.tools:
        args.tools = json.loads(args.tools)
    else:
        args.tools = None

    main(args)
