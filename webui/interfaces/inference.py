from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import json
import asyncio
from threading import Thread
import torch
from concurrent.futures import ThreadPoolExecutor

from ..configs.logging_config import LOGGING_CONFIG
from ..configs.model_config import MODEL_CONFIG, MODEL_GENERATION_CONFIG
from ..configs.model_lists import BASE_MODEL_LIST, INSTRUCT_MODEL_LIST, CUSTOM_MODEL_LIST
from ..utils.db_utils import MessagesModel


# 配置日志记录
logging.basicConfig(**LOGGING_CONFIG)
# 设置日志记录器
logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=4)

# 数据模型
class InferenceRequest(BaseModel):
    """消息请求模型"""
    model_name_or_path: str
    conversation: list
    generation_config: Optional[dict] = None
    documents: Optional[str] = None
    tools: Optional[list] = None
    feedback: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None
    notes: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    device_map: Optional[str] = None
    torch_dtype: Optional[str] = None


# 推理函数
async def async_iterator_wrapper(sync_iterator):
    """将同步迭代器转换为异步迭代器"""
    for item in sync_iterator:
        await asyncio.sleep(0)  # 允许其他异步任务运行
        yield item

async def stream_generate(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    conversation: list,
    documents: Optional[list] = None,
    tools: Optional[list] = None,
    generation_config: Optional[dict] = None,
    device: Optional[torch.device] = None,
):
    """
    使用模型生成流式回答
    
    Args:
        tokenizer (AutoTokenizer): 模型的tokenizer
        model (AutoModelForCausalLM): 模型
        conversation (list): 对话历史 [{"role": "user", "content": "问题内容"}, {"role": "assistant", "content": "回答内容"}]
        documents (Optional[list], optional): 相关文档, 默认为None. [{"title": "文档标题", "text": "文档内容"}]
        tools (Optional[list], optional): 工具列表, 默认为None. [{"name": "工具名称", "description": "工具描述", "parameters": {"param1": "value1", "param2": "value2"}}]
        generation_config (Optional[dict], optional): 生成配置, 默认为None. {"max_length": 512, "temperature": 0.7, "top_p": 0.9, "top_k": 50}
    """
    generated_content = ""
    generation_task = None

    try:
        if model is None or tokenizer is None:
            yield json.dumps({
                "content": "模型未正确加载, 无法提供回答.", 
                "done": True
            })
            return
        
        # 检查并设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用模型的chat template来处理对话
        inputs = tokenizer.apply_chat_template(
            conversation=conversation,
            documents=documents,
            tools=tools,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        # 确保输入张量在正确的设备上
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # 设置模型到评估模式并确保在正确的设备上
        model.eval()
        if next(model.parameters()).device != device:
            model.to(device)
    
        # 设置流式生成的参数, 如果没有提供则使用配置文件中的默认值
        generation_config = generation_config or MODEL_GENERATION_CONFIG.copy()

        # 创建streamer
        streamer = TextIteratorStreamer(
            tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True, 
            timeout=5.0
        )
        
        # 在后台线程中运行生成
        generation_kwargs = dict(
            inputs=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            streamer=streamer,
            **generation_config
        )

        # 使用ThreadPoolExecutor来运行生成任务
        loop = asyncio.get_event_loop()
        generation_task = loop.run_in_executor(
            executor, 
            lambda: model.generate(**generation_kwargs)
        )
        
        # 流式输出
        async for new_content in async_iterator_wrapper(streamer):
            generated_content += new_content
            yield json.dumps({
                "message": generated_content,
                "done": False
            })
        
        # 等待生成任务完成
        try:
            await asyncio.wait_for(generation_task, timeout=60.0)  # 设置合理的超时时间
        except asyncio.TimeoutError:
            logging.warning("模型生成超时")
            yield json.dumps({
                "content": generated_content,
                "error": "生成超时",
                "done": True}
            )
            return
        
        yield json.dumps({"content": generated_content, "done": True})
        
        # 主动清理缓存以释放GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        # logger.error(f"生成回答时出错: {str(e)}", exc_info=True)
        error_msg = str(e) if len(str(e)) < 200 else str(e)[:200] + "..."
        yield json.dumps({
            "content": generated_content, 
            "error": f"生成过程发生错误: {error_msg}", 
            "done": True
        })
        
        # 出现错误时尝试主动清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 消息处理函数
def messages_inference(
    conversation: list,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    model_name_or_path: str,
    documents: Optional[list] = None,
    tools: Optional[list] = None,
    generation_config: Optional[dict] = None,
    device: Optional[torch.device] = None,
    feedback: Optional[str] = None,
    source: Optional[str] = None,
    score: Optional[float] = None,
    notes: Optional[str] = None,
    message_model: Optional[MessagesModel] = None,
):
    """
    使用模型生成回答, 并在完成生成后更新消息记录
    """

    async def answer_generator():
        async for chunk in stream_generate(
            tokenizer=tokenizer,
            model=model,
            conversation=conversation,
            documents=documents,
            tools=tools,
            generation_config=generation_config,
            device=device
        ):
            # 处理生成的chunk
            chunk_data = json.loads(chunk)
            
            # 如果有错误信息, 则记录错误
            if "error" in chunk_data:
                yield json.dumps({
                    "content": chunk_data["content"],
                    "error": chunk_data["error"],
                    "done": True
                })
                return
            
            # 更新消息记录
            if chunk_data["done"]:
                message_model.create(
                    message_json=chunk_data["content"],
                    model_path_or_name=model_name_or_path,
                    feedback=feedback,
                    source=source,
                    score=score,
                    notes=notes
                )

            yield json.dumps({
                "content": chunk_data["content"],
                "done": True
            })

    return StreamingResponse(
        answer_generator(),
        media_type="application/json",
        headers={"Content-Type": "application/json"}
    )


def load_model(
    model_name_or_path: str,
    trust_remote_code: Optional[bool] = None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> dict:
    """加载模型"""
    if model_name_or_path in (BASE_MODEL_LIST + INSTRUCT_MODEL_LIST + CUSTOM_MODEL_LIST):
        model_config = MODEL_CONFIG.copy()
        model_config['model_name_or_path'] = model_name_or_path # 更新模型名称或路径
        model_config['trust_remote_code'] = trust_remote_code if trust_remote_code is not None else model_config['trust_remote_code']
        model_config['device_map'] = device_map if device_map is not None else model_config['device_map']
        model_config['torch_dtype'] = torch_dtype if torch_dtype is not None else model_config['torch_dtype']

        try:
            # 尝试加载
            tokenizer = AutoTokenizer.from_pretrained(**model_config)
            model = AutoModelForCausalLM.from_pretrained(**model_config).eval()
            logger.info(f"加载模型 {model_config['model_name_or_path']} 成功, 使用 {model_config['device_map']} 设备, 精度类型 {model_config['torch_dtype']}")
            
            return tokenizer, model
        except Exception as e:
            logger.error(f"加载模型失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")
    else:
        logger.error(f"模型名称或路径不在可用列表中: {model_name_or_path}")
        return None, None