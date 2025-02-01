from datasets import load_dataset
from argparse import ArgumentParser

def download_smoltalk(save_dir, cache_dir, num_proc):
    # Download smoltalk dataset
    dataset = load_dataset("HuggingFaceTB/smoltalk", "all", num_proc=num_proc, cache_dir=cache_dir)
    print(dataset)
    dataset.save_to_disk(save_dir + "/smoltalk", num_proc=num_proc)

def download_ultrafeedback_binarized(save_dir, cache_dir, num_proc):
    # Download ultrafeedback_binarized dataset
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", num_proc=num_proc, cache_dir=cache_dir)
    print(dataset)
    dataset.save_to_disk(save_dir + "/ultrafeedback_binarized", num_proc=num_proc)

# You can also download other datasets

def main(args):
    download_smoltalk(args.save_dir, args.cache_dir, args.num_proc)
    download_ultrafeedback_binarized(args.save_dir, args.cache_dir, args.num_proc)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./datasets")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    main(args)
