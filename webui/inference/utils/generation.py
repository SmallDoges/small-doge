from typing import List, Dict, Tuple, Optional, Callable, Union

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
import threading



def generate_response(
    tokenizer: AutoTokenizer, # 分词器
    model: AutoModelForCausalLM, # 模型
    tools: Optional[List[Union[Dict, Callable]]] = None, # 工具列表
    documents: Optional[List[Dict[str, str]]] = None, # 文档列表
    user_message: str = "hello", # 用户消息
    history: List[Tuple[str, Optional[str]]] = [], # 历史消息
    system_prompt: str = "You are a helpful assistant.", # 系统提示
    min_new_tokens: int = 0, # 最小生成 token 数
    max_new_tokens: int = 512, # 最大生成 token 数
    temperature: float = 0.8, # 采样温度
    top_p: float = 0.9, # 采样的前 p 个 token
    top_k: int = 50, # 采样的前 k 个 token
    repetition_penalty: float = 1.0, # 重复惩罚
):
    # 构建对话历史
    conversation = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        # 检查助手消息是否为 None 或空，避免错误
        if assistant_msg:
            conversation.append({"role": "assistant", "content": assistant_msg})

    # 添加当前用户消息
    conversation.append({"role": "user", "content": user_message})

    # 准备模型输入
    inputs = tokenizer.apply_chat_template(
        conversation=conversation,
        tools=tools,
        documents=documents,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True
    )

    # 使用当前侧边栏设置创建生成配置
    generation_config = GenerationConfig(
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id, # 使用 eos_token_id
        pad_token_id=tokenizer.eos_token_id # 设置 pad_token_id 防止警告
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 在后台线程中生成文本
    generation_kwargs = dict(
        **inputs,
        tokenizer=tokenizer,
        generation_config=generation_config,
        streamer=streamer
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 流式生成文本
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer # 每次产生累积的文本