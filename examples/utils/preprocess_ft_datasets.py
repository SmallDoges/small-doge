import re
from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict
from argparse import ArgumentParser
import trl


# In the case of TRL>=0.15.0, apply_chat_template does not need to be applied here, but in the case of TRL<0.15.0, apply_chat_template needs to be applied here
if trl.__version__ >= "0.15.0":
    is_apply_chat_template = False
else:
    is_apply_chat_template = True


def smoltalk_map(example, tokenizer):
    messages = example['messages']

    if is_apply_chat_template:
        example['text'] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    else:
        example['text'] = messages

    return example

def ultrafeedback_binarized_map(example, tokenizer):
        
    prompt_messages = example["chosen"][:-1]
    chosen_messages = example["chosen"][-1:]
    rejected_messages = example["rejected"][-1:]

    if is_apply_chat_template:
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
    else:
        example['text_prompt'] = prompt_messages
        example['text_chosen'] = chosen_messages
        example['text_rejected'] = rejected_messages

    return example

def open_thoughts_map(example, tokenizer):
    # prompt = example['system']
    messages = example['conversations']

    cleaned_assistant = re.sub(
        r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>', '', 
        messages[1]["value"],
        flags=re.DOTALL
    )
    messages[1]["value"] = cleaned_assistant
    conversations = [
        # {"role": "system", "content": prompt},
        {"role": "user", "content": messages[0]["value"]},
        {"role": "assistant", "content": messages[1]["value"]},
    ]

    if is_apply_chat_template:
        example['messages'] = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
        )
    else:
        example['messages'] = conversations
    return example

def openr1_math_map(example):
    system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <|begin_of_thought|> <|end_of_thought|> and <|begin_of_solution|> <|end_of_solution|> tags, respectively, i.e., <|begin_of_thought|> reasoning process here <|end_of_thought|><|begin_of_solution|> answer here <|end_of_solution|>."
    messages = example["messages"]

    # Replace <think> with <|begin_of_thought|>, </think> with <|end_of_thought|>, <answer> with <|begin_of_solution|>, </answer> with <|end_of_solution|>
    for message in messages:
        message["content"] = message["content"].replace("<think>", "<|begin_of_thought|>").replace("</think>", "<|end_of_thought|>").replace("<answer>", "<|begin_of_solution|>").replace("</answer>", "<|end_of_solution|>")
    example["messages"] = messages

    return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["problem"]},
            ],
        }

def huatuo_map(example):
    system_prompt = "你是一个医学助手，能够回答用户提出的医学问题。请根据用户的问题，给出准确的医学建议和解答。"
    
    questions = example["questions"]
    answer = example["answers"][0] if isinstance(example["answers"], list) else example["answers"]

    primary_question = questions[0][0]
    user_content = primary_question

    if len(questions) > 1 or (len(questions) == 1 and len(questions[0]) > 1):
        user_content = "、".join([q for sublist in questions for q in sublist])
    
    conversations = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]

    example["messages"] = conversations
    return example



def process_smoltalk(tokenizer, datasets_dir, num_proc):
    dataset = load_from_disk(datasets_dir + '/smoltalk')
    columns = dataset['train'].column_names
    dataset = dataset.map(
        smoltalk_map, 
        fn_kwargs={
            'tokenizer': tokenizer
        },
        num_proc=num_proc,
        remove_columns=columns,
        desc="Processing smoltalk"
    )
    print(dataset)
    dataset.save_to_disk(datasets_dir + '/smoltalk_processed')
    return "smoltalk_processed"

def process_ultrafeedback_binarized(tokenizer, datasets_dir, num_proc):
    dataset = load_from_disk(datasets_dir + '/ultrafeedback_binarized')
    dataset = DatasetDict({
        'train': dataset['train_prefs'],
        'test': dataset['test_prefs']
    })
    columns = dataset['train'].column_names
    dataset = dataset.map(
        ultrafeedback_binarized_map, 
        fn_kwargs={
            'tokenizer': tokenizer
        },
        num_proc=num_proc,
        remove_columns=columns,
        desc="Processing ultrafeedback_binarized"
    )
    dataset = dataset.rename_columns(
        {
            'text_prompt': "prompt",
            'text_chosen': "chosen",
            'text_rejected': "rejected"
        }
    )
    print(dataset)
    dataset.save_to_disk(datasets_dir + '/ultrafeedback_binarized_processed')
    return "ultrafeedback_binarized_processed"

def process_open_thoughts(tokenizer, datasets_dir, num_proc):
    dataset = load_from_disk(datasets_dir + '/open_thoughts')
    columns = dataset['train'].column_names
    dataset = dataset.map(
        open_thoughts_map, 
        fn_kwargs={
            'tokenizer': tokenizer
        },
        num_proc=num_proc,
        remove_columns=columns,
        desc="Processing open_thoughts"
    )
    print(dataset)
    dataset.save_to_disk(datasets_dir + '/open_thoughts_processed')
    return "open_thoughts_processed"

def process_openr1_math(tokenizer, datasets_dir, num_proc):
    dataset = load_from_disk(datasets_dir + '/openr1_math')
    dataset = dataset.map(
        openr1_math_map,
        num_proc=num_proc,
        desc="Processing openr1_math"
    )
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    print(dataset)
    dataset.save_to_disk(datasets_dir + '/openr1_math_processed')
    return "openr1_math_processed"

def process_huatuo(tokenizer, datasets_dir, num_proc):
    dataset = load_from_disk(datasets_dir + '/huatuo_encyclopedia_qa')
    columns = dataset['train'].column_names if 'train' in dataset else dataset.column_names
    dataset = dataset.map(
        huatuo_map,
        num_proc=num_proc,
        remove_columns=columns,
        desc="Processing huatuo"
    )
    
    if is_apply_chat_template:
        dataset = dataset.map(
            lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)},
            num_proc=num_proc,
            desc="Applying chat template to huatuo"
        )
    else:
        dataset = dataset.map(
            lambda x: {"text": x["messages"]},
            num_proc=num_proc,
            desc="Converting messages to text format for huatuo"
        )
    
    print(dataset)
    dataset.save_to_disk(datasets_dir + '/huatuo_processed')
    return "huatuo_processed"


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Process smoltalk dataset
    process_smoltalk(tokenizer, args.datasets_dir, args.num_proc)

    # Process ultrafeedback_binarized dataset
    process_ultrafeedback_binarized(tokenizer, args.datasets_dir, args.num_proc)

    # Process open-thoughts dataset
    process_open_thoughts(tokenizer, args.datasets_dir, args.num_proc)

    # Process openr1-math dataset
    process_openr1_math(tokenizer, args.datasets_dir, args.num_proc)

    # Process huatuo dataset
    process_huatuo(tokenizer, args.datasets_dir, args.num_proc)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--tokenizer_name_or_path", type=str, default="SmallDoge/Doge-tokenizer")
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)
