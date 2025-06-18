---
library_name: transformers
datasets:
- HuggingFaceTB/smollm-corpus
---

# Doge-tokenizer
 Tokenizer for the training model on [smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), and support reasoning fine-tuning like R1.
This tokenizer was trained on 2M samples from:
 - FineWeb-Edu 70%
 - Cosmopedia v2 20%
 - Python-Edu  5%
 - FineMath 5%

## How to use

<details>
<summary>Only conversation</summary>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")

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
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2024
Today Date: December 2025
You are a helpful assistant.<|end_of_text|>

<|start_header_id|>user<|end_header_id|>
What's wrong with Cheems? What treatment does he need?<|end_of_text|>

<|start_header_id|>assistant<|end_header_id|>
Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.<|end_of_text|>
```
</details>

<details>
<summary>With documents</summary>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")

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
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2024
Today Date: December 2025
You are a helpful assistant.<|end_of_text|>

<|start_header_id|>user<|end_header_id|>
Given the following documents, please use them to answer the user's question.

Title: Cheems' case
Content: Cheems is a 233-year-old alchemist, but he is very kidney deficient.

If the documents don't contain relevant information, rely on your general knowledge but acknowledge when you're doing so.

What's wrong with Cheems? What treatment does he need?<|end_of_text|>

<|start_header_id|>assistant<|end_header_id|>
Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.<|end_of_text|>
```
</details>

<details>
<summary>With tools</summary>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")

tools = [
    {
        "name": "recommend_treatment",
        "description": "Treatment is recommended based on patient symptoms and diagnosis.",
        "parameters": {
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
    {"role": "assistant", "tool_calls": [
        {
            "id": "call_1",
            "function": {
                "name": "recommend_treatment",
                "arguments": "{\"diagnosis\": \"Deficiency of kidney\", \"severity\": \"severe\"}"
            }
        }
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."},
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
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Environment: ipython
Cutting Knowledge Date: December 2024
Today Date: December 2025
You are a helpful assistant.<|end_of_text|>

<|start_header_id|>user<|end_header_id|>
Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

{
    "name": "recommend_treatment",
    "description": "Treatment is recommended based on patient symptoms and diagnosis.",
    "parameters": {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": "Patient diagnostic results"
            },
            "severity": {
                "type": "string",
                "enum": [
                    "mild",
                    "moderate",
                    "severe"
                ],
                "description": "Severity of the condition, can be mild, moderate, or severe."
            }
        },
        "required": [
            "diagnosis",
            "severity"
        ]
    }
}

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

What's wrong with Cheems? What treatment does he need?<|end_of_text|>

<|start_header_id|>assistant<|end_header_id|>
{"name": "recommend_treatment", "parameters": "{\"diagnosis\": \"Deficiency of kidney\", \"severity\": \"severe\"}"}<|end_of_text|>

<|start_header_id|>ipython<|end_header_id|>
"For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."<|end_of_text|>

<|start_header_id|>assistant<|end_header_id|>
Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.<|end_of_text|>
```
</details>

<details>
<summary>With documents and tools</summary>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SmallDoge/Doge-tokenizer")

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
        "parameters": {
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
    {"role": "assistant", "tool_calls": [
        {
            "id": "call_1",
            "function": {
                "name": "recommend_treatment",
                "arguments": "{\"diagnosis\": \"Deficiency of kidney\", \"severity\": \"severe\"}"
            }
        }
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."},
    {"role": "assistant", "content": "Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."},
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
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Environment: ipython
Cutting Knowledge Date: December 2024
Today Date: December 2025
You are a helpful assistant.<|end_of_text|>

<|start_header_id|>user<|end_header_id|>
Given the following documents, please use them to answer the user's question.

Title: Cheems' case
Content: Cheems is a 233-year-old alchemist, but he is very kidney deficient.

If the documents don't contain relevant information, rely on your general knowledge but acknowledge when you're doing so.

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

{
    "name": "recommend_treatment",
    "description": "Treatment is recommended based on patient symptoms and diagnosis.",
    "parameters": {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": "Patient diagnostic results"
            },
            "severity": {
                "type": "string",
                "enum": [
                    "mild",
                    "moderate",
                    "severe"
                ],
                "description": "Severity of the condition, can be mild, moderate, or severe."
            }
        },
        "required": [
            "diagnosis",
            "severity"
        ]
    }
}

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

What's wrong with Cheems? What treatment does he need?<|end_of_text|>

<|start_header_id|>assistant<|end_header_id|>
{"name": "recommend_treatment", "parameters": "{\"diagnosis\": \"Deficiency of kidney\", \"severity\": \"severe\"}"}<|end_of_text|>

<|start_header_id|>ipython<|end_header_id|>
"For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters."<|end_of_text|>

<|start_header_id|>assistant<|end_header_id|>
Based on Cheems' symptoms, I recommend the following treatment: For Cheems with severe kidney deficiency, it is recommended to eat more leeks and oysters.<|end_of_text|>
```
</details>