import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr


from configs import (
    BASE_MODEL_LIST,
    INSTRUCT_MODEL_LIST,
    DEFAULT_GENERATION_CONFIG,
)
from utils import (
    generate_response
)


# 全局变量，用于存储当前加载的模型和分词器
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 模型加载函数 ---
def load_model(model_name):
    global tokenizer, model
    
    # 释放之前加载的模型内存（如果有）
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 加载新选择的模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    return f"模型 {model_name} 加载成功！"


# --- Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), title="SmallDoge") as demo:
    # 状态变量
    chat_history = gr.State([])
    left_sidebar_visible = gr.State(False)
    right_sidebar_visible = gr.State(False)
    current_model = gr.State("SmallDoge/Doge-320M-Instruct")  # 默认模型
    model_load_status = gr.State("")

    # --- 顶部按钮行 ---
    with gr.Row():
        toggle_left_btn = gr.Button("☰ 模型选择", size="sm")
        gr.Markdown("# SmallDoge")
        toggle_right_btn = gr.Button("⚙️ 参数设置", size="sm")

    # --- 主布局行 ---
    with gr.Row(equal_height=False):
        # --- 左侧边栏 (模型选择) ---
        with gr.Column(scale=1, visible=False) as left_sidebar:
            gr.Markdown("## 选择模型")
            
            # 模型类型选择器
            model_type = gr.Radio(
                ["基础模型", "指令微调模型"],
                label="模型类型",
                value="指令微调模型"
            )
            
            # 根据模型类型动态更新模型列表
            model_selector = gr.Dropdown(
                choices=INSTRUCT_MODEL_LIST,  # 默认显示指令微调模型
                label="选择模型",
                value=INSTRUCT_MODEL_LIST[0] if INSTRUCT_MODEL_LIST else None
            )
            
            # 模型加载状态显示
            model_status = gr.Markdown("")
            
            # 加载模型按钮
            load_model_btn = gr.Button("加载选中模型", variant="primary")
            
            gr.Markdown("---")
            system_prompt_input = gr.Textbox(
                value="You are a helpful assistant.",
                label="系统提示 (System Prompt)",
                lines=3
            )

        # --- 中间聊天区域 ---
        with gr.Column(scale=4):
            chatbot_display = gr.Chatbot(
                label="聊天窗口",
                bubble_full_width=False,
                height=600 # 设置聊天窗口高度
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="输入你的消息...",
                    scale=4,
                    show_label=False,
                    container=False # 使其与按钮在同一行更好看
                )
                submit_btn = gr.Button("发送", variant="huggingface", scale=1)

            # 添加清除聊天按钮
            clear_btn = gr.Button("清除聊天", scale=1)

            # 示例问题按钮 (可选)
            with gr.Row():
                 gr.Examples(
                    examples=[
                        "你好，请介绍一下自己。",
                        "What's wrong with Cheems? What treatment does he need?",
                        "写一个关于小狗的笑话"
                    ],
                    inputs=[user_input],
                    label="示例问题"
                 )


        # --- 右侧边栏 (参数设置) ---
        with gr.Column(scale=1, visible=False) as right_sidebar:
            gr.Markdown("## 生成参数设置")
            with gr.Accordion("基本参数", open=True):
                min_tokens = gr.Number(value=DEFAULT_GENERATION_CONFIG["min_new_tokens"], label="最小生成token数", precision=0)
                max_tokens = gr.Number(value=DEFAULT_GENERATION_CONFIG["max_new_tokens"], label="最大生成token数", precision=0)
                temp = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=DEFAULT_GENERATION_CONFIG["temperature"], label="Temperature")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=DEFAULT_GENERATION_CONFIG["top_p"], label="Top-p")
                top_k_slider = gr.Slider(minimum=1, maximum=100, step=1, value=DEFAULT_GENERATION_CONFIG["top_k"], label="Top-k")
                rep_penalty = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=DEFAULT_GENERATION_CONFIG["repetition_penalty"], label="重复惩罚")

    # --- 事件处理 ---

    # 根据模型类型更新模型选择器
    def update_model_list(model_type_value):
        if model_type_value == "基础模型":
            return gr.update(choices=BASE_MODEL_LIST, value=BASE_MODEL_LIST[0] if BASE_MODEL_LIST else None)
        else:  # 指令微调模型
            return gr.update(choices=INSTRUCT_MODEL_LIST, value=INSTRUCT_MODEL_LIST[0] if INSTRUCT_MODEL_LIST else None)
    
    # 绑定模型类型选择事件
    model_type.change(
        update_model_list,
        inputs=[model_type],
        outputs=[model_selector]
    )
    
    # 加载模型事件处理
    def handle_model_load(model_name):
        status_message = load_model(model_name)
        return model_name, status_message
    
    # 绑定加载模型按钮
    load_model_btn.click(
        handle_model_load,
        inputs=[model_selector],
        outputs=[current_model, model_status]
    )

    # 侧边栏切换函数
    def toggle_sidebar_visibility(current_visibility):
        return gr.update(visible=not current_visibility)

    # 绑定左侧边栏切换按钮
    toggle_left_btn.click(
        toggle_sidebar_visibility,
        inputs=[left_sidebar_visible],
        outputs=[left_sidebar],
    ).then(lambda x: not x, inputs=left_sidebar_visible, outputs=left_sidebar_visible) # 更新状态

    # 绑定右侧边栏切换按钮
    toggle_right_btn.click(
        toggle_sidebar_visibility,
        inputs=[right_sidebar_visible],
        outputs=[right_sidebar],
    ).then(lambda x: not x, inputs=right_sidebar_visible, outputs=right_sidebar_visible) # 更新状态

    # 处理用户提交
    def handle_submit(message, history, sys_prompt, min_t, max_t, t, top_p, top_k, rep_p):
        # 更新历史记录 (用户消息)
        history.append([message, None]) # 添加用户消息，AI响应暂时为None
        return gr.update(value=""), history, history

    # 处理流式响应
    def handle_stream(history, message, sys_prompt, min_t, max_t, t, top_p, top_k, rep_p):
        global tokenizer, model  # 确保可以访问全局变量
        
        # 正确顺序传递参数
        stream = generate_response(
            tokenizer=tokenizer,  
            model=model,
            tools=None,  
            documents=None,
            user_message=message,
            history=history,
            system_prompt=sys_prompt,
            min_new_tokens=min_t,
            max_new_tokens=max_t,
            temperature=t,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=rep_p
        )

        if not history:
            yield history
            return

        # 流式更新最后一个助手的消息
        history[-1][1] = ""  # 初始化AI响应为空字符串
        for response_chunk in stream:
            history[-1][1] = response_chunk  # 更新最后一个AI响应
            yield history  # 每次更新都产生新的聊天记录状态
    
    def clear_chat_history():
        return [], []  # 返回空列表分别更新chat_history和chatbot_display

    # 绑定提交事件
    submit_inputs = [user_input, chat_history, system_prompt_input, min_tokens, max_tokens, temp, top_p_slider, top_k_slider, rep_penalty]
    submit_outputs = [user_input, chat_history, chatbot_display]
    clear_btn.click(
        clear_chat_history,
        outputs=[chat_history, chatbot_display]
    )
    
    # handle_stream 需要的输入： history (来自 submit_outputs[1]), message (来自 submit_inputs[0]), 和其他参数 (来自 submit_inputs[2:])
    stream_inputs = [chat_history, user_input, system_prompt_input, min_tokens, max_tokens, temp, top_p_slider, top_k_slider, rep_penalty]

    # 使用 .then() 来链式处理：先提交，再流式响应
    user_input.submit(
        handle_submit,
        inputs=submit_inputs,
        outputs=submit_outputs,
        queue=False # 先快速完成提交处理
    ).then(
        handle_stream,
        inputs=stream_inputs, # 传递正确的参数给流式处理
        outputs=[chatbot_display] # 流式更新聊天窗口
    )

    submit_btn.click(
        handle_submit,
        inputs=submit_inputs,
        outputs=submit_outputs,
        queue=False
    ).then(
        handle_stream,
        inputs=stream_inputs, # 传递正确的参数给流式处理
        outputs=[chatbot_display]
    )

    # 初始加载默认模型
    demo.load(
        lambda: load_model("SmallDoge/Doge-320M-Instruct")
    )

# 启动Gradio服务
if __name__ == "__main__":
    demo.queue().launch(share=True) # 使用 queue() 以更好地处理并发和流式输出