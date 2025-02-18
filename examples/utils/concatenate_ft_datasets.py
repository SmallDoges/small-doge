from datasets import concatenate_datasets, load_from_disk, Dataset, DatasetDict
from argparse import ArgumentParser


def concatenate_sft_datasets(datasets_dir, save_dir, num_proc):
    smoltalk_dataset = load_from_disk(datasets_dir + '/smoltalk_processed')
    train_dataset : Dataset = concatenate_datasets([
        smoltalk_dataset['train'],
    ])
    test_dataset : Dataset = concatenate_datasets([
        smoltalk_dataset['test'],
    ])
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    dataset = dataset.shuffle(seed=233)
    dataset.save_to_disk(save_dir + "/sft_dataset", num_proc=num_proc, num_shards={'train': 16, 'test': 1 })

def concatenate_dpo_datasets(datasets_dir, save_dir, num_proc):
    ultrafeedback_binarized_dataset = load_from_disk(datasets_dir + '/ultrafeedback_binarized_processed')
    train_dataset : Dataset = concatenate_datasets([
        ultrafeedback_binarized_dataset['train'],
    ])
    test_dataset : Dataset = concatenate_datasets([
        ultrafeedback_binarized_dataset['test'],
    ])
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    dataset = dataset.shuffle(seed=233)
    dataset.save_to_disk(save_dir + "/dpo_dataset", num_proc=num_proc, num_shards={'train': 16, 'test': 1 })

def concatenate_distill_datasets(datasets_dir, save_dir, num_proc):
    open_thoughts_dataset = load_from_disk(datasets_dir + '/open_thoughts_processed')
    train_dataset : Dataset = concatenate_datasets([
        open_thoughts_dataset['train'],
    ])
    dataset = train_dataset.train_test_split(test_size=1_000, shuffle=True, seed=233)
    dataset.save_to_disk(save_dir + "/distill_dataset", num_proc=num_proc, num_shards={'train': 16, 'test': 1 })

def concatenate_grpo_datasets(datasets_dir, save_dir, num_proc):
    openr1_math_dataset = load_from_disk(datasets_dir + '/openr1_math_processed')
    train_dataset : Dataset = concatenate_datasets([
        openr1_math_dataset['train'],
    ])
    dataset = train_dataset.train_test_split(test_size=1_000, shuffle=True, seed=233)
    dataset.save_to_disk(save_dir + "/grpo_dataset", num_proc=num_proc, num_shards={'train': 16, 'test': 1 })


def main(args):

    # Concatenate sft datasets
    concatenate_sft_datasets(args.datasets_dir, args.save_dir, args.num_proc)

    # Concatenate dpo datasets
    concatenate_dpo_datasets(args.datasets_dir, args.save_dir, args.num_proc)

    # Concatenate distill datasets
    concatenate_distill_datasets(args.datasets_dir, args.save_dir, args.num_proc)

    # Concatenate grpo datasets
    concatenate_grpo_datasets(args.datasets_dir, args.save_dir, args.num_proc)


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)