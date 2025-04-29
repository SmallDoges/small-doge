import os
import json
import logging
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
from argparse import ArgumentParser


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_json_file(file_path):
    """Load JSON configuration file"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_json_file(data, file_path):
    """Save JSON configuration file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_training_corpus(dataset):
    """Generate training corpus in batches"""
    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]["text"]


def ensure_all_special_tokens(old_tokenizer_path, new_tokenizer, new_tokenizer_dir):
    """Ensure all special tokens are correctly added to the new tokenizer"""
    logger.info("Ensuring all special tokens are correctly added...")
    
    # Load original added_tokens.json directly
    old_added_tokens_path = os.path.join(old_tokenizer_path, "added_tokens.json")
    if not os.path.exists(old_added_tokens_path):
        logger.warning(f"Warning: added_tokens.json not found in {old_tokenizer_path}")
        return
        
    old_added_tokens = load_json_file(old_added_tokens_path)
    
    # Load new added_tokens.json
    new_added_tokens_path = os.path.join(new_tokenizer_dir, "added_tokens.json")
    new_added_tokens = load_json_file(new_added_tokens_path) if os.path.exists(new_added_tokens_path) else {}
    
    # Check for missing tokens
    missing_tokens = [token for token, _ in old_added_tokens.items() if token not in new_added_tokens]
    
    if missing_tokens:
        logger.info(f"Found {len(missing_tokens)} missing special tokens, adding: {missing_tokens}")
        
        # Add missing tokens to new tokenizer
        new_tokenizer.add_tokens(missing_tokens, special_tokens=True)
        
        # Update added_tokens.json
        updated_added_tokens = {
            token: new_tokenizer.convert_tokens_to_ids(token) 
            for token in old_added_tokens 
            if token in new_tokenizer.get_vocab()
        }
        
        # Save updated added_tokens.json
        save_json_file(updated_added_tokens, new_added_tokens_path)
        
        logger.info(f"Successfully added {len(missing_tokens)} missing special tokens")
    else:
        logger.info("No missing special tokens found")


def update_config_files(old_tokenizer_path, new_tokenizer_dir, new_tokenizer):
    """Update tokenizer configuration files"""
    # Handle tokenizer_config.json
    old_config_path = os.path.join(old_tokenizer_path, "tokenizer_config.json")
    new_config_path = os.path.join(new_tokenizer_dir, "tokenizer_config.json")
    
    if os.path.exists(old_config_path) and os.path.exists(new_config_path):
        old_config = load_json_file(old_config_path)
        new_config = load_json_file(new_config_path)
        
        # Update vocabulary size
        new_config["vocab_size"] = len(new_tokenizer)
        
        # Keep model_max_length consistent
        if "model_max_length" in old_config:
            new_config["model_max_length"] = old_config["model_max_length"]
        
        # Ensure special token information is consistent
        special_token_keys = [
            "bos_token", "eos_token", "unk_token", "sep_token", 
            "pad_token", "cls_token", "mask_token"
        ]
        for key in special_token_keys:
            if key in old_config:
                new_config[key] = old_config[key]
        
        # Save updated configuration
        save_json_file(new_config, new_config_path)
    
    # Ensure special_tokens_map.json is consistent
    old_map_path = os.path.join(old_tokenizer_path, "special_tokens_map.json")
    new_map_path = os.path.join(new_tokenizer_dir, "special_tokens_map.json")
    
    if os.path.exists(old_map_path):
        old_map = load_json_file(old_map_path)
        save_json_file(old_map, new_map_path)


def validate_tokenizer(old_tokenizer, new_tokenizer, test_texts):
    """Validate consistency between old and new tokenizers"""
    logger.info("Validating consistency between old and new tokenizers...")
    
    # Check special token count
    old_added_tokens = old_tokenizer.get_added_vocab()
    new_added_tokens = new_tokenizer.get_added_vocab()
    logger.info(f"Original tokenizer special tokens count: {len(old_added_tokens)}")
    logger.info(f"New tokenizer special tokens count: {len(new_added_tokens)}")
    
    # Check missing tokens
    missing = [token for token in old_added_tokens if token not in new_added_tokens]
    
    if missing:
        logger.warning(f"Warning: New tokenizer is missing these special tokens: {missing}")
    else:
        logger.info("All special tokens have been properly preserved")
    
    # Check encoding consistency
    for text in test_texts:
        old_encoding = old_tokenizer.encode(text)
        new_encoding = new_tokenizer.encode(text)
        truncated_text = text[:30] + '...' if len(text) > 30 else text
        logger.info(f"Test text: '{truncated_text}'")
        logger.info(f"  Old tokenizer tokens count: {len(old_encoding)}")
        logger.info(f"  New tokenizer tokens count: {len(new_encoding)}")


def main(args):
    logger.info(f"Loading dataset: {args.dataset_name_or_path}")
    try:
        dataset = load_dataset(
            args.dataset_name_or_path,
            split="train",
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    if args.num_examples and args.num_examples < len(dataset):
        logger.info(f"Selecting {args.num_examples} examples for training")
        dataset = dataset.select(range(args.num_examples))
    
    logger.info(f"Loading original tokenizer: {args.old_tokenizer_path}")
    try:
        old_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.old_tokenizer_path)
    except Exception as e:
        logger.error(f"Failed to load original tokenizer: {e}")
        return
    
    # Get all special token information
    added_tokens = old_tokenizer.get_added_vocab()
    additional_special_tokens = old_tokenizer.additional_special_tokens
    
    logger.info(f"Original vocabulary size: {len(old_tokenizer)}")
    logger.info(f"Original special tokens count: {len(added_tokens)}")
    logger.info(f"Original additional_special_tokens count: {len(additional_special_tokens)}")
    
    # Prepare training corpus
    training_corpus = get_training_corpus(dataset)
    
    # Prepare all special tokens as new_special_tokens parameter
    all_special_tokens = list(added_tokens.keys())
    
    # Train new tokenizer
    logger.info(f"Starting to train new tokenizer, target vocabulary size: {args.vocab_size}")
    try:
        new_tokenizer = old_tokenizer.train_new_from_iterator(
            training_corpus, 
            vocab_size=args.vocab_size,
            new_special_tokens=all_special_tokens
        )
    except Exception as e:
        logger.error(f"Failed to train new tokenizer: {e}")
        return
    
    logger.info(f"New vocabulary size: {len(new_tokenizer)}")
    
    # Ensure new tokenizer directory exists
    os.makedirs(args.new_tokenizer_save_dir, exist_ok=True)
    
    # Save new tokenizer first
    new_tokenizer.save_pretrained(args.new_tokenizer_save_dir)
    
    # Ensure all special tokens are correctly added
    ensure_all_special_tokens(args.old_tokenizer_path, new_tokenizer, args.new_tokenizer_save_dir)
    
    # Update configuration files
    update_config_files(args.old_tokenizer_path, args.new_tokenizer_save_dir, new_tokenizer)
    
    # Reload updated tokenizer to validate
    updated_tokenizer = AutoTokenizer.from_pretrained(args.new_tokenizer_save_dir)
    
    # Validate new tokenizer
    test_texts = [
        "This is a test text for validating tokenizer functionality.",
        "def hello_world():\n    print('Hello, world!')",
        "<|im_start|>This is a text with special tokens<|im_end|>",
        "<think>Thinking process</think>",
        "<tool_call>Tool call</tool_call>"
    ]
    validate_tokenizer(old_tokenizer, updated_tokenizer, test_texts)
    
    logger.info(f"New tokenizer saved to: {args.new_tokenizer_save_dir}")
    logger.info(f"New tokenizer special tokens count: {len(updated_tokenizer.get_added_vocab())}")


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_name_or_path", type=str, default="SmallDoge/MiniCorpus", help="Dataset name or path")
    argparser.add_argument("--old_tokenizer_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Original tokenizer path")
    argparser.add_argument("--new_tokenizer_save_dir", type=str, default="./tokenizer_new", help="Directory to save new tokenizer")
    argparser.add_argument("--num_examples", type=int, default=2_000_000, help="Number of examples to use for training")
    argparser.add_argument("--vocab_size", type=int, default=32768, help="Vocabulary size for new tokenizer")
    argparser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to cache dataset")
    args = argparser.parse_args()

    main(args)