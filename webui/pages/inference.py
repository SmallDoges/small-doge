import gradio as gr
import time
import os
import sys
import json
from datetime import datetime

from ..configs.logging_config import LOGGING_CONFIG
from ..utils.api_client import ApiClient
import logging


# 配置日志
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class InferencePage:
    def __init__(self, api_client: ApiClient):
        self.api_client = api_client
    
    def handle_user_message(self, message, conversation):
        """处理用户消息"""
        return "", conversation + [{"role": "user", "content": message}]
    
    def handle_assistant_message(self, conversation):
        """处理助手消息"""
        try:
            for response in self.api_client.inference(conversation):
                try:
                    if response["status"] == "success":
        
    def create_inference_page(self):
        with gr.Blocks(title="SmallDoge") as inference_page:
            gr.Markdown("# SmallDoge Inference Page")
            
            # 消息状态变量
            message = gr.ChatMessage()

            # 聊天窗口
            with gr.Column(visible=True) as chat_window:
                chatbot = gr.Chatbot(
                    label="Chatbot",
                    type="messages",
                )
            
            # 输入框
            with gr.Row():
                with gr.Column(scale=0.85):
                    input_text = gr.Textbox(
                        label="Input",
                        placeholder="Type your message here...",
                        show_label=False,
                        lines=1,
                    )
                
                with gr.Column(scale=0.15):
                    submit_button = gr.Button("🤗", variant="huggingface")
                
            # 清除按钮
            clear_button = gr.Button("Clear", variant="primary")

            # 页脚
            gr.Markdown("---")
