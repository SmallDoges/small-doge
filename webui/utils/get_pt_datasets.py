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
from dataclasses import dataclass, field
from typing import List, Optional, Union
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer
from argparse import ArgumentParser


logger = logging.getLogger(__name__)


@dataclass
class PTDatasetsArguments:
    tokenizer_name_or_path: str = field(metadata={"help": "The name or path of the tokenizer to use."})
    datasets_name_and_ratio: List[dict] = field(metadata={"help": "The name or path of the dataset to download, and the ratio of each dataset for mixed final dataset."})
    split: str = field(default="all", metadata={"help": "The split to download."})
    save_dir: str = field(default="./datasets", metadata={"help": "The directory to save the dataset."})
    cache_dir: str = field(default="./cache", metadata={"help": "The directory to cache the dataset."})
    seed: int = field(default=233, metadata={"help": "The seed to use."})
    num_proc: int = field(default=1, metadata={"help": "The number of processes to use."})
    max_length: int = field(default=2048, metadata={"help": "The maximum length of the text."})
    truncation: bool = field(default=True, metadata={"help": "Whether to truncate the text."})
    padding: bool = field(default=False, metadata={"help": "Whether to pad the text."})
    train_examples: int = field(default=128_000_000, metadata={"help": "The number of training examples."})
    test_examples: int = field(default=1_000, metadata={"help": "The number of testing examples."})


def get_pt_datasets(args: PTDatasetsArguments):
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # process map
    def process_map(example, tokenizer, max_length=args.max_length, truncation=args.truncation, padding=args.padding):
        prompt = example["prompt"] if "prompt" in example else None
        text = example["text"]
        outputs = tokenizer(
            prompt,
            text,
            add_special_tokens=True,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }
    
    datasets = []
    for dataset_name_and_ratio in args.datasets_name_and_ratio:
        
        # get the dataset name or path and ratio
        dataset_name_or_path = list(dataset_name_and_ratio.keys())[0]
        dataset_ratio = list(dataset_name_and_ratio.values())[0]

        # load the dataset
        dataset = load_dataset(dataset_name_or_path, split=args.split, num_proc=args.num_proc, cache_dir=args.cache_dir)

        # process the dataset
        column_names = dataset.column_names
        dataset_len = len(dataset)
        example_len = (args.train_examples + args.test_examples) * dataset_ratio
        if dataset_len < example_len:
            raise ValueError(f"The dataset {dataset_name_or_path} does not have enough examples. {dataset_len} < {example_len}")
        dataset = dataset.shuffle(seed=args.seed).select(range(example_len))
        dataset = dataset.map(
            process_map, 
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": args.max_length,
                "truncation": args.truncation,
                "padding": args.padding
            },
            num_proc=args.num_proc,
            remove_columns=column_names,
            batched=True,
            desc=f"Processing {dataset_name_or_path}"
        )

        datasets.append(dataset)

    # concatenate all dataset
    datasets: Dataset = concatenate_datasets(datasets)

    # split train and test sets and shuffle
    datasets = datasets.train_test_split(test_size=args.test_examples, shuffle=True, seed=args.seed)

    # save dataset
    datasets.save_to_disk(f"{args.save_dir}", num_proc=args.num_proc)

# # 测试get_pt_datasets
# args = PTDatasetsArguments(
#     tokenizer_name_or_path="SmallDoge/Doge-tokenizer",
#     datasets_name_and_ratio=[{"HuggingFaceTB/cosmopedia-20k": 0.5}, {"HuggingFaceTB/cosmopedia-20k": 0.5}],
#     split="all", 
#     save_dir="datasets/pt_dataset", 
#     cache_dir="cache", 
#     num_proc=1
#     )
# get_pt_datasets(args)
