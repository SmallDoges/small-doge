from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict
from argparse import ArgumentParser


def smoltalk_map(example, tokenizer):
    messages = example['messages']
    example['text'] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )
    return example

def ultrafeedback_binarized_map(example, tokenizer):
        
    prompt_messages = example["chosen"][:-1]
    chosen_messages = example["chosen"][-1:]
    rejected_messages = example["rejected"][-1:]

    example['text_prompt'] = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
    )
    example['text_chosen'] = tokenizer.apply_chat_template(
        chosen_messages,
        tokenize=False,
    )
    example['text_rejected'] = tokenizer.apply_chat_template(
        rejected_messages,
        tokenize=False,
    )

    return example

def bespoke_stratos_map(example, tokenizer):
    messages = example['messages']
    example['text'] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )
    return example

SYSTEM_PROMPT_FOR_GRPO = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def numinamath_map(example):
    return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_FOR_GRPO},
                {"role": "user", "content": example["problem"]},
            ],
        }


def process_smoltalk(tokenizer, datasets_dir, num_proc):
    dataset = load_from_disk(datasets_dir + '/smoltalk')
    columns = dataset['train'].column_names
    dataset = dataset.map(
        smoltalk_map, 
        fn_kwargs={
            'tokenizer': tokenizer
        },
        num_proc=num_proc,
        remove_columns=columns,
        desc="Processing smoltalk"
    )
    print(dataset)
    dataset.save_to_disk(datasets_dir + '/smoltalk_processed')
    return "smoltalk_processed"

def process_ultrafeedback_binarized(tokenizer, datasets_dir, num_proc):
    dataset = load_from_disk(datasets_dir + '/ultrafeedback_binarized')
    dataset = DatasetDict({
        'train': dataset['train_prefs'],
        'test': dataset['test_prefs']
    })
    columns = dataset['train'].column_names
    dataset = dataset.map(
        ultrafeedback_binarized_map, 
        fn_kwargs={
            'tokenizer': tokenizer
        },
        num_proc=num_proc,
        remove_columns=columns,
        desc="Processing ultrafeedback_binarized"
    )
    dataset = dataset.rename_columns(
        {
            'text_prompt': "prompt",
            'text_chosen': "chosen",
            'text_rejected': "rejected"
        }
    )
    print(dataset)
    dataset.save_to_disk(datasets_dir + '/ultrafeedback_binarized_processed')
    return "ultrafeedback_binarized_processed"

def process_bespoke_stratos(tokenizer, datasets_dir, num_proc):
    dataset = load_from_disk(datasets_dir + '/bespoke_stratos')
    columns = dataset['train'].column_names
    dataset = dataset.map(
        bespoke_stratos_map, 
        fn_kwargs={
            'tokenizer': tokenizer
        },
        num_proc=num_proc,
        remove_columns=columns,
        desc="Processing bespoke_stratos"
    )
    print(dataset)
    dataset.save_to_disk(datasets_dir + '/bespoke_stratos_processed')
    return "bespoke_stratos_processed"

def process_numinamath(tokenizer, datasets_dir, num_proc):
    dataset = load_from_disk(datasets_dir + '/numinamath')
    columns = dataset['train'].column_names
    dataset = dataset.map(
        numinamath_map, 
        num_proc=num_proc,
        remove_columns=columns,
        desc="Processing numinamath"
    )
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    print(dataset)
    dataset.save_to_disk(datasets_dir + '/numinamath_processed')
    return "numinamath_processed"


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Process smoltalk dataset
    process_smoltalk(tokenizer, args.datasets_dir, args.num_proc)

    # Process ultrafeedback_binarized dataset
    process_ultrafeedback_binarized(tokenizer, args.datasets_dir, args.num_proc)

    # Process bespoke_stratos dataset
    process_bespoke_stratos(tokenizer, args.datasets_dir, args.num_proc)

    # Process numinamath dataset
    process_numinamath(tokenizer, args.datasets_dir, args.num_proc)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--tokenizer_name_or_path", type=str, default="SmallDoge/Doge-tokenizer")
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)
