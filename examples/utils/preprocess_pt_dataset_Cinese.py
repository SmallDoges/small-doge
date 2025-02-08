from transformers import AutoTokenizer
# from dataset import load_from_disk
from argparse import ArgumentParser
from datasets import load_from_disk

from datasets import load_dataset

dataset = load_dataset('json', data_files='./dataset/mobvoi_seq_monkey_general_open_corpus.jsonl', split='train', num_proc=8, cache_dir='./dataset')


def process_monkey_general(example, tokenizer, max_length=2048):
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
    tokenizer = AutoTokenizer.from_pretrained(args.mtokenizer_path)
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.map(
        lambda x: process_monkey_general(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset.save_to_disk(args.output_path)

if __name__ == '__main__':
    dataset.save_to_disk('./dataset', num_proc=8)
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--mtokenizer_path', type=str, default='./tokenizer_new')
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()
    main(args)

