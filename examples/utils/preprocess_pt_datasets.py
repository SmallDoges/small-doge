from transformers import AutoTokenizer
from datasets import load_from_disk
from argparse import ArgumentParser


def process_fineweb_edu(example, tokenizer, max_length=2048):
    text = example['text']
    outputs = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        'input_ids': outputs['input_ids'],
        'attention_mask': outputs['attention_mask'],
    }

def process_cosmopedia(example, tokenizer, max_length=2048):
    prompt = example['prompt']
    text = example['text']
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text},
    ]
    outputs = tokenizer.apply_chat_template(
        conversation, 
        tokenize=True, 
        truncation=True, 
        padding=False, 
        max_length=max_length,
        return_overflowing_tokens=False,
        return_length=False,
        return_dict=True
    )
    return {
        'input_ids': outputs['input_ids'],
        'attention_mask': outputs['attention_mask'],
    }

def process_python_edu(example, tokenizer, max_length=2048):
    text = example['text']
    outputs = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        'input_ids': outputs['input_ids'],
        'attention_mask': outputs['attention_mask'],
    }

def process_fine_math(example, tokenizer, max_length=2048):
    text = example['text']
    outputs = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        'input_ids': outputs['input_ids'],
        'attention_mask': outputs['attention_mask'],
    }

def main(args):

    # Calculate the size of fineweb-edu, cosmopedia-v2, python-edu, fine-math
    fineweb_edu_ratio, cosmopedia_v2_ratio, python_edu_ratio, fine_math_ratio = 0.7, 0.2, 0.05, 0.05

    fineweb_edu_train_size = int(args.train_examples * fineweb_edu_ratio)
    cosmopedia_v2_train_size = int(args.train_examples * cosmopedia_v2_ratio)
    python_edu_train_size = int(args.train_examples * python_edu_ratio)
    fine_math_train_size = int(args.train_examples * fine_math_ratio)

    fineweb_edu_test_size = int(args.test_examples * fineweb_edu_ratio)
    cosmopedia_v2_test_size = int(args.test_examples * cosmopedia_v2_ratio)
    python_edu_test_size = int(args.test_examples * python_edu_ratio)
    fine_math_test_size = int(args.test_examples * fine_math_ratio)


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Process fineweb-edu
    dataset = load_from_disk(args.datasets_dir + '/fineweb-edu')
    column_names = dataset.column_names
    dataset = dataset.select(
        range(fineweb_edu_train_size + fineweb_edu_test_size - 1, -1, -1)
    ).map(
        process_fineweb_edu, 
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': args.max_length
        },
        num_proc=args.num_proc,
        remove_columns=column_names,
        batched=True,
        desc="Processing fineweb-edu"
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/fineweb-edu_processed')

    # Process Cosmopedia-v2
    dataset = load_from_disk(args.datasets_dir + '/cosmopedia-v2')
    column_names = dataset.column_names

    dataset = dataset.select(
        range(cosmopedia_v2_train_size + cosmopedia_v2_test_size)
    ).map(
        process_cosmopedia, 
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': args.max_length
        },
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Processing cosmopedia-v2"
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/cosmopedia-v2_processed')

    # Process Python Education
    dataset = load_from_disk(args.datasets_dir + '/python-edu')
    column_names = dataset.column_names
    dataset = dataset.select(
        range(python_edu_train_size + python_edu_test_size)
    ).map(
        process_python_edu, 
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': args.max_length
        },
        num_proc=args.num_proc,
        remove_columns=column_names,
        batched=True,
        desc="Processing python-edu"
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/python-edu_processed')

    # Process FineMath
    dataset = load_from_disk(args.datasets_dir + '/finemath')
    column_names = dataset.column_names
    dataset = dataset.select(
        range(fine_math_train_size + fine_math_test_size)
    ).map(
        process_fine_math,
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': args.max_length
        },
        num_proc=args.num_proc,
        remove_columns=column_names,
        batched=True,
        desc="Processing finemath"
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/finemath_processed')

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--tokenizer_name_or_path", type=str, default="SmallDoge/Doge-tokenizer")
    argparser.add_argument("--train_examples", type=int, default=128_000_000)
    argparser.add_argument("--test_examples", type=int, default=1_000)
    argparser.add_argument("--max_length", type=int, default=2048)
    argparser.add_argument("--num_proc", type=int, default=1)
    args = argparser.parse_args()

    main(args)
