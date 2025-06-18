---
library_name: transformers
datasets:
- SmallDoge/SmallCorpus
---

# Doge2-tokenizer

Tokenizer for the training model on [SmallCorpus](https://huggingface.co/datasets/SmallDoge/SmallCorpus), supporting document retrieval, tool invocation, and reasoning.
This tokenizer was trained on 2M samples from:
- Web-EN 50%
- Web-ZH 20%
- TextBook-EN 15%
- TextBook-ZH 5%
- Code 5%
- Math 5%

## How to use

<details>
<summary>Only conversation</summary>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge2-tokenizer")

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's wrong with Cheems? What treatment does he need?"},
    {"role": "assistant", "content": "Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."},
]

inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    tokenize=False,
)
print(inputs)
```
```shell
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What's wrong with Cheems? What treatment does he need?<|im_end|>
<|im_start|>assistant
<think>

</think>

Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.<|im_end|>

```
</details>

<details>
<summary>With documents</summary>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge2-tokenizer")

documents = [
    {
        "title": "Cheems's case",
        "text": "Cheems is a 233-year-old alchemist, but he is very kidney deficient."
    },
]

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's wrong with Cheems? What treatment does he need?"},
    {"role": "assistant", "content": "Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."},
]

inputs = tokenizer.apply_chat_template(
    documents=documents,
    conversation=conversation,
    tokenize=False,
)
print(inputs)
```
```shell
<|im_start|>system
You are a helpful assistant.

You have access to the following documents. Please use them to answer the user's question.

Title: Cheems's case
Content: Cheems is a 233-year-old alchemist, but he is very kidney deficient.

If the documents don't contain relevant information, rely on your general knowledge but acknowledge when you're doing so.<|im_end|>
<|im_start|>user
What's wrong with Cheems? What treatment does he need?<|im_end|>
<|im_start|>assistant
<think>

</think>

Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.<|im_end|>
```
</details>

<details>
<summary>With tools</summary>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge2-tokenizer")

tools = [
    {
        "name": "recommend_treatment",
        "description": "Treatment is recommended based on patient symptoms and diagnosis.",
        "arguments": {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "Patient diagnostic results"
                },
                "severity": {
                    "type": "string",
                    "enum": ["mild", "moderate", "severe"],
                    "description": "Severity of the condition, can be mild, moderate, or severe."
                }
            },
            "required": ["diagnosis", "severity"]
        }
    }
]

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's wrong with Cheems? What treatment does he need?"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"recommend_treatment\", \"arguments\": {\"diagnosis\": \"Deficiency of kidney\", \"severity\": \"severe\"}}\n</tool_call>"},
    {"role": "tool", "content": "For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."},
    {"role": "assistant", "content": "Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."},
]

inputs = tokenizer.apply_chat_template(
    tools=tools,
    conversation=conversation,
    tokenize=False,
)
print(inputs)
```
```shell
<|im_start|>system
You are a helpful assistant.

You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags:

<tools>
{"name": "recommend_treatment", "description": "Treatment is recommended based on patient symptoms and diagnosis.", "arguments": {"type": "object", "properties": {"diagnosis": {"type": "string", "description": "Patient diagnostic results"}, "severity": {"type": "string", "enum": ["mild", "moderate", "severe"], "description": "Severity of the condition, can be mild, moderate, or severe."}}, "required": ["diagnosis", "severity"]}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What's wrong with Cheems? What treatment does he need?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "recommend_treatment", "arguments": {"diagnosis": "Deficiency of kidney", "severity": "severe"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.<|im_end|>
```
</details>

<details>
<summary>With reasoning</summary>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge2-tokenizer")

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's wrong with Cheems? What treatment does he need?"},
    {"role": "assistant", "reasoning_content": "Cheems is a dog, a old dog. He is been around for a long time, so his innards are not as good as before. If he is not careful, he will get kidney deficiency. He needs to eat more leeks and oysters.", "content": "Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."},
]

inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    tokenize=False,
)
print(inputs)
```
```shell
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What's wrong with Cheems? What treatment does he need?<|im_end|>
<|im_start|>assistant
<think>
Cheems is a dog, a old dog. He is been around for a long time, so his innards are not as good as before. If he is not careful, he will get kidney deficiency. He needs to eat more leeks and oysters.
</think>

Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.<|im_end|>
```
</details>

<details>
<summary>With documents tools reasoning</summary>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge2-tokenizer")

documents = [
    {
        "title": "Cheems's case",
        "text": "Cheems is a 233-year-old alchemist, but he is very kidney deficient."
    },
]

tools = [
    {
        "name": "recommend_treatment",
        "description": "Treatment is recommended based on patient symptoms and diagnosis.",
        "arguments": {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "Patient diagnostic results"
                },
                "severity": {
                    "type": "string",
                    "enum": ["mild", "moderate", "severe"],
                    "description": "Severity of the condition, can be mild, moderate, or severe."
                }
            },
            "required": ["diagnosis", "severity"]
        }
    }
]

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's wrong with Cheems? What treatment does he need?"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"recommend_treatment\", \"arguments\": {\"diagnosis\": \"Deficiency of kidney\", \"severity\": \"severe\"}}\n</tool_call>"},
    {"role": "tool", "content": "For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."},
    {"role": "assistant", "reasoning_content": "Cheems is a dog, a old dog. He is been around for a long time, so his innards are not as good as before. If he is not careful, he will get kidney deficiency. He needs to eat more leeks and oysters.", "content": "Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."},
]

inputs = tokenizer.apply_chat_template(
    documents=documents,
    tools=tools,
    conversation=conversation,
    tokenize=False,
)
print(inputs)
```
```shell
<|im_start|>system
You are a helpful assistant.

You have access to the following documents. Please use them to answer the user's question.

Title: Cheems's case
Content: Cheems is a 233-year-old alchemist, but he is very kidney deficient.

If the documents don't contain relevant information, rely on your general knowledge but acknowledge when you're doing so.

You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags:

<tools>
{"name": "recommend_treatment", "description": "Treatment is recommended based on patient symptoms and diagnosis.", "arguments": {"type": "object", "properties": {"diagnosis": {"type": "string", "description": "Patient diagnostic results"}, "severity": {"type": "string", "enum": ["mild", "moderate", "severe"], "description": "Severity of the condition, can be mild, moderate, or severe."}}, "required": ["diagnosis", "severity"]}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What's wrong with Cheems? What treatment does he need?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "recommend_treatment", "arguments": {"diagnosis": "Deficiency of kidney", "severity": "severe"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.
</tool_response><|im_end|>
<|im_start|>assistant
<think>
Cheems is a dog, a old dog. He is been around for a long time, so his innards are not as good as before. If he is not careful, he will get kidney deficiency. He needs to eat more leeks and oysters.
</think>

Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.<|im_end|>
```
</details>