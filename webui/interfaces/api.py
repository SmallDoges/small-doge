from fastapi import FastAPI, Depends, HTTPException, Query, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .inference import messages_inference, load_model, InferenceRequest
from ..configs.logging_config import LOGGING_CONFIG
from ..configs.api_config import FASTAPI_SERVER_CONFIG, CORS_CONFIG, UNICORN_CONFIG
from ..configs.model_config import MODEL_CONFIG, MODEL_GENERATION_CONFIG
from ..configs.model_lists import BASE_MODEL_LIST, INSTRUCT_MODEL_LIST, CUSTOM_MODEL_LIST
from ..utils.db_utils import DatabaseConnection, MessagesModel


# 配置日志记录
logging.basicConfig(**LOGGING_CONFIG)
# 设置日志记录器
logger = logging.getLogger(__name__)


# 初始化FastAPI应用
app = FastAPI(**FASTAPI_SERVER_CONFIG)

# 添加CORS中间件
app.add_middleware(CORSMiddleware, **CORS_CONFIG)

# 初始化数据库连接
db_connection = DatabaseConnection()
messages_model = MessagesModel(db_connection)

# 加载LLM
TOKENIZER = AutoTokenizer.from_pretrained(**MODEL_CONFIG)
MODEL = AutoModelForCausalLM.from_pretrained(**MODEL_CONFIG).eval()

logger.info(f"加载模型 {MODEL_CONFIG['model_name']}, 使用 {MODEL_CONFIG['device']} 设备, 精度类型 {MODEL_CONFIG['dtype']}")

# === API路由 ===

# 消息推理路由
@app.post("/api/inference")
async def inference_route(request: InferenceRequest):
    """消息推理接口"""
    if request.model_name_or_path not in (BASE_MODEL_LIST + INSTRUCT_MODEL_LIST + CUSTOM_MODEL_LIST):
        logger.warning(f"模型 {request.model_name_or_path} 不在可用模型列表中, 继续使用默认模型")
        pass
    else:
        # 检查模型是否已经加载
        if request.model_name_or_path != MODEL_CONFIG["model_name_or_path"]:
            # 如果模型名称不同，则重新加载模型
            logger.info(f"重新加载模型 {request.model_name_or_path}")
            TOKENIZER, MODEL = load_model(
                model_name_or_path=request.model_name_or_path,
                trust_remote_code=request.trust_remote_code,
                device_map=request.device_map,
                torch_dtype=request.torch_dtype
            )
            if TOKENIZER is None or MODEL is None:
                raise HTTPException(status_code=500, detail="模型加载失败")
        else:
            logger.info(f"使用已加载的模型 {request.model_name_or_path}")
    
    return messages_inference(
        conversation=request.conversation,
        tokenizer=TOKENIZER,
        model=MODEL,
        model_name_or_path=request.model_name_or_path,
        documents=request.documents,
        tools=request.tools,
        generation_config=MODEL_GENERATION_CONFIG,
        device=request.device_map,
        feedback=request.feedback,
        source=request.source,
        score=request.score,
        notes=request.notes,
        messages_model=messages_model
    )

# @app.post("/api/public_info/set")
# async def set_public_status(request: SetPublicInfoRequest):
#     return set_public_info(
#         request,
#         question_model=question_model,
#         answer_model=answer_model,
#         public_info_model=public_info_model
#     )

# @app.get("/api/public_info/list")
# async def list_public_info(
#     status: str = Query("all", description="状态筛选: all, pending, approved, rejected"),
#     page: int = Query(1, ge=1, description="页码"),
#     page_size: int = Query(10, ge=1, le=100, description="每页记录数")
# ):
#     """管理员获取公开信息列表"""
#     # TODO: 添加管理员权限验证
#     # 这里简化处理，实际应检查当前用户是否是管理员
    
#     offset = (page - 1) * page_size
    
#     # 连接数据库
#     if not db_connection.connect():
#         raise HTTPException(status_code=500, detail="数据库连接失败")
    
#     try:
#         # 构建查询SQL
#         if status == "all":
#             query = """
#             SELECT p.*, u.Username, q.QuestionContent, a.AnswerContent 
#             FROM PublicInformation p
#             JOIN User u ON p.UserID = u.UserID
#             JOIN Question q ON p.QuestionID = q.QuestionID
#             JOIN Answer a ON p.AnswerID = a.AnswerID
#             ORDER BY p.SubmissionDate DESC
#             LIMIT %s OFFSET %s
#             """
#             count_query = "SELECT COUNT(*) as count FROM PublicInformation"
#             params = (page_size, offset)
#             count_params = ()
#         else:
#             query = """
#             SELECT p.*, u.Username, q.QuestionContent, a.AnswerContent 
#             FROM PublicInformation p
#             JOIN User u ON p.UserID = u.UserID
#             JOIN Question q ON p.QuestionID = q.QuestionID
#             JOIN Answer a ON p.AnswerID = a.AnswerID
#             WHERE p.Status = %s
#             ORDER BY p.SubmissionDate DESC
#             LIMIT %s OFFSET %s
#             """
#             count_query = "SELECT COUNT(*) as count FROM PublicInformation WHERE Status = %s"
#             params = (status, page_size, offset)
#             count_params = (status,)
        
#         # 执行查询
#         cursor = db_connection.execute_query(query, params)
#         records = cursor.fetchall()
        
#         # 获取总记录数
#         count_cursor = db_connection.execute_query(count_query, count_params)
#         total_count = count_cursor.fetchone()["count"]
        
#         # 转换记录格式
#         result_records = []
#         for record in records:
#             result_records.append({
#                 "public_information_id": record["PublicInformationID"],
#                 "user_id": record["UserID"],
#                 "username": record["Username"],
#                 "question_content": record["QuestionContent"],
#                 "answer_content": record["AnswerContent"],
#                 "submission_date": record["SubmissionDate"].isoformat(),
#                 "status": record["Status"],
#                 "notes": record["Notes"]
#             })
        
#         return {
#             "status": "success",
#             "total_records": total_count,
#             "records": result_records
#         }
#     except Exception as e:
#         logging.error(f"查询公开信息列表失败: {e}")
#         raise HTTPException(status_code=500, detail=f"查询公开信息列表失败: {str(e)}")
#     finally:
#         db_connection.disconnect()

# @app.put("/api/public_info/{public_info_id}")
# async def update_public_info_status(
#     public_info_id: int = Path(..., description="公开信息ID"),
#     update_data: PublicInfoStatusUpdate = None
# ):
#     """更新公开信息状态（管理员）"""
#     # TODO: 添加管理员权限验证
    
#     # 检查公开信息是否存在
#     public_info = public_info_model.get_by_id(public_info_id)
#     if not public_info:
#         raise HTTPException(status_code=404, detail="公开信息不存在")
    
#     # 验证状态值
#     if update_data.status not in ["pending", "approved", "rejected"]:
#         raise HTTPException(status_code=400, detail="无效的状态值")
    
#     # 更新状态
#     updated = public_info_model.update_status(
#         public_info_id=public_info_id,
#         status=update_data.status,
#         notes=update_data.notes
#     )
    
#     if not updated:
#         raise HTTPException(status_code=500, detail="更新公开信息状态失败")
    
#     return {
#         "status": "success",
#         "message": f"公开信息状态已更新为 {update_data.status}"
#     }

# @app.post("/api/knowledge_base/add")
# async def add_knowledge_route(request: AddKnowledgeRequest):
#     """添加知识库条目"""
#     # TODO: 添加管理员权限验证
#     return add_knowledge(
#         request,
#         knowledge_base_model=knowledge_base_model
#     )

# @app.post("/api/knowledge_base/add_from_public_info")
# async def add_from_public_info_route(request: AddFromPublicInfoRequest):
#     """从公开信息添加到知识库和训练集"""
#     # TODO: 添加管理员权限验证
#     return add_from_public_info(
#         request,
#         public_info_model=public_info_model,
#         question_model=question_model,
#         answer_model=answer_model,
#         knowledge_base_model=knowledge_base_model,
#         training_data_model=training_data_model
#     )

# @app.get("/api/knowledge_base/search")
# async def search_knowledge_route(
#     query: str = Query(..., description="查询文本"), 
#     top_k: int = Query(5, ge=1, le=50, description="返回结果数量")
# ):
#     """搜索知识库"""
#     return search_knowledge(
#         query=query,
#         top_k=top_k,
#         knowledge_base_model=knowledge_base_model
#     )

# # 模型训练路由
# @app.post("/api/model/train")
# async def train_model_route(request: TrainingRequest, background_tasks: BackgroundTasks):
#     """启动模型训练"""
#     # TODO: 添加管理员权限验证
#     return await start_training(
#         request=request,
#         background_tasks=background_tasks,
#         tokenizer=TOKENIZER,
#         model=MODEL,
#         training_data_model=training_data_model
#     )

# @app.get("/api/model/status/{task_id}")
# async def get_model_training_status(task_id: str):
#     """获取训练状态"""
#     # TODO: 添加管理员权限验证
#     return get_training_status(task_id)

# @app.post("/api/model/stop")
# async def stop_model_training(request: TaskIDRequest):
#     """停止模型训练"""
#     # TODO: 添加管理员权限验证
#     return stop_training(request.task_id)

# @app.get("/api/model/list")
# async def list_model_training_tasks():
#     """列出所有训练任务"""
#     # TODO: 添加管理员权限验证
#     return list_training_tasks()

# @app.get("/api/model/load/{task_id}")
# async def load_model_route(task_id: str):
#     """加载训练完成的模型"""
#     # TODO: 添加管理员权限验证
#     result = load_trained_model(
#         task_id=task_id
#     )
#     TOKENIZER = result["tokenizer"]
#     MODEL = result["model"]
#     return {
#         "status": "success",
#         "message": f"模型 {task_id} 已加载"
#     }

# 启动应用
if __name__ == "__main__":
    uvicorn.run(app, **UNICORN_CONFIG)