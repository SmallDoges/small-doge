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
import io
import os
import sys
from typing import List, Optional, Union, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, Response

import datasets
import transformers

from get_pt_datasets import PTDatasetsArguments, get_pt_datasets

logger = logging.getLogger(__name__)

# Setup logging
io_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("./app.log")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[io_handler, file_handler],
)
logger.setLevel(logging.INFO)
datasets.utils.logging.set_verbosity(logging.INFO)
datasets_logger = datasets.utils.logging.get_logger()
datasets_logger.addHandler(file_handler)
transformers.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.get_logger()
transformers.utils.logging.enable_explicit_format()


# Define the API router
router = APIRouter(
    tags=["small-doge"],
    responses={404: {"description": "Not found"}},
)


class PTDatasetsArgumentsRequest(BaseModel):
    tokenizer_name_or_path: str
    datasets_name_and_ratio: List[Dict[str, float]]
    split: str = "all"
    save_dir: str = "./datasets"
    cache_dir: str = "./cache"
    seed: int = 233
    num_proc: int = 1
    max_length: int = 2048
    truncation: bool = True
    padding: bool = False
    train_examples: int = 128_000_000
    test_examples: int = 1_000


@router.post("/get_pt_datasets")
async def get_pt_datasets_api(args: PTDatasetsArgumentsRequest):
    # Convert the request to the dataclass
    args = PTDatasetsArguments(**args.model_dump())

    # Get the datasets
    get_pt_datasets(args)

    return Response(content=f"Datasets saved to {args.save_dir}")


# Define the APP
app = FastAPI(
    title="SmallDoge API",
    description="API for SmallDoge",
    version="0.1.0",
)

# Include the router
app.include_router(router)


# Run the APP
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)