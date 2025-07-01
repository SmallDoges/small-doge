#!/usr/bin/env python3
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

"""
Unified training entry script
Automatically select corresponding model type and trainer based on recipe path
"""

import os
import sys
import argparse

def get_trainer_type_from_args():
    """Infer trainer type from command line arguments"""
    # Check common training type identifiers
    script_name = os.path.basename(sys.argv[0])
    
    if 'pt' in script_name or 'pretrain' in script_name:
        return 'pt'
    elif 'sft' in script_name or 'finetune' in script_name:
        return 'sft'
    elif 'dpo' in script_name:
        return 'dpo'
    elif 'grpo' in script_name:
        return 'grpo'
    
    # Check config parameter
    for arg in sys.argv:
        if arg.startswith('--config'):
            if '=' in arg:
                config_path = arg.split('=')[1]
            else:
                # Find next argument as config path
                try:
                    idx = sys.argv.index(arg)
                    config_path = sys.argv[idx + 1]
                except (ValueError, IndexError):
                    continue            
            # Infer type from config path
            config_path = config_path.lower()
            if 'pt' in config_path or 'pretrain' in config_path:
                return 'pt'
            elif 'sft' in config_path or 'finetune' in config_path:
                return 'sft'
            elif 'dpo' in config_path:
                return 'dpo'
            elif 'grpo' in config_path:
                return 'grpo'
    # Default return pt
    return 'pt'


def main():
    """Unified training entry point"""
    
    # Parse basic arguments to determine training type
    parser = argparse.ArgumentParser(description="Unified SmallDoge Trainer")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--trainer_type', type=str, choices=['pt', 'sft', 'dpo', 'grpo'], 
                       help='Explicitly specify trainer type')
    # Only parse known arguments, pass the rest to specific trainer
    known_args, remaining_args = parser.parse_known_args()
    
    # Determine trainer type
    trainer_type = known_args.trainer_type
    if not trainer_type:
        trainer_type = get_trainer_type_from_args()
    
    print(f"Using trainer type: {trainer_type}")    
    # Import and run corresponding trainer
    try:
        if trainer_type == 'pt':
            from small_doge.trainer.pt import main as trainer_main
            # Re-parse arguments for pt trainer
            from argparse import ArgumentParser
            import yaml
            from trl import ModelConfig, ScriptArguments, TrlParser
            from transformers import TrainingArguments
            
            model_config_parser = ArgumentParser()
            model_config_parser.add_argument(
                "--config", type=str, required=True, help="path to yaml config file"
            )
            
            parser = TrlParser((ScriptArguments, TrainingArguments, ModelConfig))
            script_args, training_args, model_args = parser.parse_args_and_config()
            
            config_path = model_config_parser.parse_args().config
            model_config = yaml.load(
                open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader
            )["model_config"]
            
            trainer_main(script_args, training_args, model_args, model_config, config_path)
            
        elif trainer_type == 'sft':
            from small_doge.trainer.sft import main as trainer_main, SFTConfig
            from trl import ModelConfig, ScriptArguments, TrlParser
            
            parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
            script_args, training_args, model_args = parser.parse_args_and_config()
            recipe_path = getattr(known_args, 'config', '')
            trainer_main(script_args, training_args, model_args, recipe_path)
            
        elif trainer_type == 'dpo':
            from small_doge.trainer.dpo import main as trainer_main, DPOConfig
            from trl import ModelConfig, ScriptArguments, TrlParser
            
            parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
            script_args, training_args, model_args = parser.parse_args_and_config()
            recipe_path = getattr(known_args, 'config', '')
            trainer_main(script_args, training_args, model_args, recipe_path)

        else:
            print(f"Unknown trainer type: {trainer_type}")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Failed to import trainer for type {trainer_type}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running trainer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
