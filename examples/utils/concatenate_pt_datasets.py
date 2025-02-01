from datasets import concatenate_datasets, load_from_disk, Dataset
from argparse import ArgumentParser

def main(args):

    # Concatenate pretraining datasets
    fineweb_edu_dataset = load_from_disk(args.datasets_dir + '/fineweb-edu_processed')
    cosmopedia_v2_dataset = load_from_disk(args.datasets_dir + '/cosmopedia-v2_processed')
    python_edu_dataset = load_from_disk(args.datasets_dir + '/python-edu_processed')
    fine_math_dataset = load_from_disk(args.datasets_dir + '/finemath_processed')
    dataset : Dataset = concatenate_datasets([
        fineweb_edu_dataset,
        cosmopedia_v2_dataset,
        python_edu_dataset,
        fine_math_dataset
    ])

    # Split train and test sets and shuffle
    dataset = dataset.train_test_split(train_size=args.train_examples, test_size=args.test_examples, shuffle=True, seed=233)

    # Save dataset
    dataset.save_to_disk(args.save_dir + "/pt_dataset", num_proc=args.num_proc, num_shards={'train': 128_000, 'test': 1 })


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--train_examples", type=int, default=128_000_000)
    argparser.add_argument("--test_examples", type=int, default=1_000)
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)