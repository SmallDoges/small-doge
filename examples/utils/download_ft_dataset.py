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

def download_bespoke_stratos(save_dir, cache_dir, num_proc):
    # Download bespoke_stratos dataset
    dataset = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", num_proc=num_proc, cache_dir=cache_dir)
    print(dataset)
    dataset.save_to_disk(save_dir + "/bespoke_stratos", num_proc=num_proc)

def download_numinamath(save_dir, cache_dir, num_proc):
    # Download numinamath dataset
    dataset = load_dataset("AI-MO/NuminaMath-TIR", num_proc=num_proc, cache_dir=cache_dir)
    print(dataset)
    dataset.save_to_disk(save_dir + "/numinamath", num_proc=num_proc)

# You can also download other datasets

def main(args):

    # For Instruction fine-tuning
    download_smoltalk(args.save_dir, args.cache_dir, args.num_proc)
    download_ultrafeedback_binarized(args.save_dir, args.cache_dir, args.num_proc)

    # For Reasoning fine-tuning
    download_bespoke_stratos(args.save_dir, args.cache_dir, args.num_proc)
    download_numinamath(args.save_dir, args.cache_dir, args.num_proc)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./datasets")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    main(args)
