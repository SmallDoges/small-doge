from datasets import concatenate_datasets, load_from_disk, Dataset
from argparse import ArgumentParser

def main(args):

    # Concatenate sft datasets
    smoltalk_dataset = load_from_disk(args.datasets_dir + '/smoltalk_processed')
    dataset : Dataset = concatenate_datasets([
        smoltalk_dataset,
    ])

    # Shuffle
    dataset = dataset.shuffle(seed=233)

    # Save dataset
    dataset.save_to_disk(args.save_dir + "/sft_dataset", num_proc=args.num_proc, num_shards={'train': 16, 'test': 1 })

    # Concatenate dpo datasets
    ultrafeedback_binarized_dataset = load_from_disk(args.datasets_dir + '/ultrafeedback_binarized_processed')
    dataset : Dataset = concatenate_datasets([
        ultrafeedback_binarized_dataset,
    ])

    # Shuffle
    dataset = dataset.shuffle(seed=233)

    # Save dataset
    dataset.save_to_disk(args.save_dir + "/dpo_dataset", num_proc=args.num_proc, num_shards={'train': 16, 'test': 1 })

    


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)