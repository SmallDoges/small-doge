from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict
from argparse import ArgumentParser


def process_smoltalk(example, tokenizer):
    messages = example['messages']
    example['text'] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )
    return example

def process_ultrafeedback_binarized(example, tokenizer):
        
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


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset = load_from_disk(args.datasets_dir + '/smoltalk')
    columns = dataset['train'].column_names
    dataset = dataset.map(
        process_smoltalk, 
        fn_kwargs={
            'tokenizer': tokenizer
        },
        num_proc=args.num_proc,
        remove_columns=columns,
        batched=True,
        desc="Applying chat template"
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/smoltalk_processed')

    dataset = load_from_disk(args.datasets_dir + '/ultrafeedback_binarized')
    dataset = DatasetDict({
        'train': dataset['train_prefs'],
        'test': dataset['test_prefs']
    })
    columns = dataset['train'].column_names
    dataset = dataset.map(
        process_ultrafeedback_binarized, 
        fn_kwargs={
            'tokenizer': tokenizer
        },
        num_proc=args.num_proc,
        remove_columns=columns,
        batched=False,
        desc="Applying chat template"
    )
    dataset = dataset.rename_columns(
        {
            'text_prompt': "prompt",
            'text_chosen': "chosen",
            'text_rejected': "rejected"
        }
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/ultrafeedback_binarized_processed')


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--tokenizer_name_or_path", type=str, default="SmallDoge/Doge-tokenizer")
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)
