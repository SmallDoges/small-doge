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

def download_open_thoughts(save_dir, cache_dir, num_proc):
    # Download open-thoughts dataset
    dataset = load_dataset("open-thoughts/OpenThoughts-114k", "default", num_proc=num_proc, cache_dir=cache_dir)
    print(dataset)
    dataset.save_to_disk(save_dir + "/open_thoughts", num_proc=num_proc)

def download_openr1_math(save_dir, cache_dir, num_proc):
    # Download openr1-math dataset
    dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", num_proc=num_proc, cache_dir=cache_dir)
    print(dataset)
    dataset.save_to_disk(save_dir + "/openr1_math", num_proc=num_proc)

def download_huatuo_encyclopedia_qa(save_dir, cache_dir, num_proc):
    # Download huatuo-encyclopedia-qa dataset
    dataset = load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa", num_proc=num_proc, cache_dir=cache_dir)
    print(dataset)
    dataset.save_to_disk(save_dir + "/huatuo_encyclopedia_qa", num_proc=num_proc)

# You can also download other datasets

def main(args):

    # For Instruction fine-tuning
    download_smoltalk(args.save_dir, args.cache_dir, args.num_proc)
    download_ultrafeedback_binarized(args.save_dir, args.cache_dir, args.num_proc)

    # For Reasoning fine-tuning
    download_open_thoughts(args.save_dir, args.cache_dir, args.num_proc)
    download_openr1_math(args.save_dir, args.cache_dir, args.num_proc)

    # For Medical fine-tuning
    download_huatuo_encyclopedia_qa(args.save_dir, args.cache_dir, args.num_proc)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./datasets")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    main(args)
