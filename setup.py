# coding=utf-8
# Copyright 2024 the SmallDoge team. All rights reserved.
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
from setuptools import setup, find_packages

__version__ = '0.1.0'

REQUIRED_PACKAGES = [
    'torch',
    'datasets',
    'transformers @ git+https://github.com/LoserCheems/transformers.git@support-constant-lr-with-cooldown',
    'accelerate',
    'trl',
    'sentencepiece',
]

file_path = os.path.abspath(os.path.dirname(__file__))

setup(
    name='small_doge',
    license='Apache 2.0',
    version=__version__,
    description="A Family of Dynamic Ultra-Fast Small Language Models Ready for Embodied Artificial General Intelligence!",
    long_description=open(os.path.join(file_path, 'README.md'), encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Jingze Shi',
    author_email="losercheems@gmail.com",
    url='https://github.com/SmallDoges/small-doge',
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    include_package_data=True,
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.9',
    zip_safe=False,
    keywords=['small-doge', 'doge', 'small language models', 'pytorch', 'transformers', 'trl'],
)