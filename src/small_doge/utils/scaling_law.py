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

import math

def openai_scaling_law(
    num_params: int,
    cs_loss: float,
):
    """
    Calculate the learning rate and batch size based on the number of parameters and cross-entropy loss following the OpenAI scaling law.
    Please refer https://arxiv.org/pdf/2001.08361 for more details.

    Args:
        num_params (int): Number of non-embedding parameters in the model.
        cs_loss (float): Cross-entropy loss.
    """
    lr = 3.239 * pow(10, -3) + -1.395 * pow(10, -4) * math.log(num_params)
    bs = 2e18 * pow(cs_loss, -4.7619)
    return lr, bs


def microsoft_scaling_law(
    num_params: int,
    num_tokens: int,
):
    """
    Calculate the learning rate and batch size based on the number of parameters and number of tokens following the Microsoft scaling law.
    Please refer https://arxiv.org/pdf/2409.19913 for more details.

    Args:
        num_params (int): Number of non-embedding parameters in the model.
        num_tokens (int): Number of tokens in the dataset.
    """
    lr = 1.3192 * pow(math.e, -5) * pow(num_params, -0.23) * pow(num_tokens, -0.32)
    return lr, None


def porian_scaling_law(
    num_params: int,
    num_tokens: int,
):
    """
    Calculate the learning rate and batch size based on the number of parameters and number of tokens following the Porian scaling law.
    Please refer https://arxiv.org/pdf/2406.19146 for more details.

    Args:
        num_params (int): Number of non-embedding parameters in the model.
        num_tokens (int): Number of tokens in the dataset.
    """
    lr = 3.7 * pow(num_params, -0.36)
    bs = 0.7576 * pow(num_params, 0.703)
    return lr, bs


def optimal_hyperparameters_scaling_law(
    num_params: int,
    num_tokens: int,
):
    """
    Calculate the learning rate and batch size based on the number of parameters and number of tokens following the optimal hyperparameters scaling law.
    Please refer https://arxiv.org/pdf/2503.04715 for more details.

    Args:
        num_params (int): Number of non-embedding parameters in the model.
        num_tokens (int): Number of tokens in the dataset.
    """
    lr = 1.79 * pow(num_params, -0.713) * pow(num_tokens, 0.307)
    bs = 0.58 * pow(num_tokens, 0.571)
    return lr, bs