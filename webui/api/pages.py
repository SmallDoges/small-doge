import logging
import gradio as gr
import requests
import json

logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/get_pt_datasets"


def call_get_pt_datasets(tokenizer_name_or_path, datasets_name_and_ratio, split, save_dir, cache_dir, seed, num_proc, max_length, truncation, padding, train_examples, test_examples):
    # ensure all arguments are in the correct format
    tokenizer_name_or_path = str(tokenizer_name_or_path)
    datasets_name_and_ratio = to_list_of_dicts(datasets_name_and_ratio)
    split = str(split)
    save_dir = str(save_dir)
    cache_dir = str(cache_dir)
    seed = int(seed)
    num_proc = int(num_proc)
    max_length = int(max_length)
    truncation = bool(truncation)
    padding = bool(padding)
    train_examples = int(train_examples)
    test_examples = int(test_examples)
    
    payload = {
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "datasets_name_and_ratio": datasets_name_and_ratio,
        "split": split,
        "save_dir": save_dir,
        "cache_dir": cache_dir,
        "seed": seed,
        "num_proc": num_proc,
        "max_length": max_length,
        "truncation": truncation,
        "padding": padding,
        "train_examples": train_examples,
        "test_examples": test_examples
    }
    
    response = requests.post(API_URL, json=payload)
    output = response.text
    return output
    

def to_list_of_dicts(datasets_name_and_ratio_str):
    data = json.loads(datasets_name_and_ratio_str)
    return [{k: v} for k, v in data.items()]

with gr.Blocks() as demo:
    gr.Markdown("# SmallDoge PT Datasets API")
    
    with gr.Row():
        with gr.Column():
            output = gr.Textbox(label="Output", lines=20)
        with gr.Column():
            tokenizer_name_or_path = gr.Textbox(label="Tokenizer Name or Path", value="SmallDoge/Doge-tokenizer")
            datasets_name_and_ratio = gr.Textbox(label="Datasets Name and Ratio", value='{"HuggingFaceTB/cosmopedia-20k": 0.5}')
            split = gr.Textbox(label="Split", value="all")
            save_dir = gr.Textbox(label="Save Directory", value="./datasets")
            cache_dir = gr.Textbox(label="Cache Directory", value="./cache")
            seed = gr.Number(label="Seed", value=233)
            num_proc = gr.Number(label="Number of Processes", value=4)
            max_length = gr.Number(label="Max Length", value=2048)
            truncation = gr.Checkbox(label="Truncation", value=True)
            padding = gr.Checkbox(label="Padding", value=False)
            train_examples = gr.Number(label="Train Examples", value=1_000)
            test_examples = gr.Number(label="Test Examples", value=1_000)
            
            submit_btn = gr.Button("Get PT Datasets")
            
            submit_btn.click(
                call_get_pt_datasets, 
                inputs=[
                    tokenizer_name_or_path,
                    datasets_name_and_ratio,
                    split,
                    save_dir,
                    cache_dir,
                    seed,
                    num_proc,
                    max_length,
                    truncation,
                    padding,
                    train_examples,
                    test_examples
                ],
                outputs=output,
            )
    
    gr.Markdown("## WandB Dashboard")
    wandb_html = gr.HTML("<iframe src='https://wandb.ai' width='100%' height='600'></iframe>")

demo.launch(debug=True)