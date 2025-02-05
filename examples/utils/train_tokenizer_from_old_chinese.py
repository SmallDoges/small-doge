'''
dataset resouces: http://share.mobvoi.com:5000/sharing/O91blwPkY

Download and unzip the file directly to the ./dataset directory

- dataset
    - mobvoi_seq_monkey_general_open_corpus.jsonl

- train_tokenizer_from_old.py

- tokenizer
    - tokenizer_new

- dataset_sample

'''

from transformers import AutoTokenizer
from datasets import load_from_disk, concatenate_datasets
from argparse import ArgumentParser
from datasets import load_dataset, Dataset

# make jsonl dataset to dataset
def load_translation_dataset(dataset_path, num_proc, cache_dir="./cache")
    dataset = load_dataset('json', data_files=dataset_path, split='train', num_proc=num_proc, cache_dir=cache_dir)
    dataset.save_to_disk('./dataset', num_proc=num_proc)
    dataset = dataset["train"].select(range(1_000_000))
    dataset.save_to_disk('./dataset_sample', num_proc=num_proc)
    return dataset

def main(args):

    dataset = load_translation_dataset('./dataset/mobvoi_seq_monkey_general_open_corpus.jsonl', args.num_proc)
    dataset = dataset.select(range(args.num_examples))

    def get_training_corpus():
        for i in range(0, len(dataset), 1):
            samples = dataset[i : i + 1]["text"]
            yield samples
    
    # Load tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_tokenizer_path)

    # Train tokenizer
    training_corpus = get_training_corpus()
    new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=32768)

    # Save new tokenizer
    new_tokenizer.save_pretrained(args.new_tokenizer_save_dir)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./dataset_sample")
    argparser.add_argument("--old_tokenizer_path", type=str, default="SmallDoge/Doge-tokenizer")
    argparser.add_argument("--new_tokenizer_save_dir", type=str, default="./tokenizer/tokenizer_new")
    argparser.add_argument("--num_examples", type=int, default=1_000_000)
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)

    # test function: load tokenizer and encode and decode text to prove the tokenizer is trained successfully
    tokenizer = AutoTokenizer.from_pretrained("./examples/tokenizer_new")
    inputs = "你好666"
    tokens = tokenizer.tokenize(inputs)
    encode = tokenizer.encode(inputs)
    decode = tokenizer.decode(encode)
    print(f"tokens: {tokens}, encode: {encode}, deocde: {decode}")
