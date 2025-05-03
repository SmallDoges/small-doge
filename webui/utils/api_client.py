import requests
import json
import os
import logging
import time
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, Union

from ..configs.logging_config import LOGGING_CONFIG
from ..configs.model_config import MODEL_CONFIG


# 配置日志
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# 全局单例实例
_instance = None

class ApiClient:
    """API客户端，用于与后端API通信"""
    
    def __new__(cls, *args, **kwargs):
        global _instance
        if not _instance:
            _instance = super(ApiClient, cls).__new__(cls)
            _instance._initialized = False
        return _instance
    
    def __init__(self, base_url="http://localhost:8000"):
        if self._initialized:
            return
        
        # 初始化API客户端参数
        self.base_url = base_url  # 默认API服务器地址
        self.timeout = 10  # 请求超时时间 (秒)
        self.headers = {"Content-Type": "application/json"}
        
        # 初始化API路径
        self.api_endpoints = {
            "inference": "/api/inference",
        }
        
        

        # 标记初始化完成
        self._initialized = True
    
    # 单例保证，防止多次初始化
    def __del__(self):
        """析构函数，释放资源"""
        pass  # 暂时不需要特殊资源释放
    
    def make_request(self, method, endpoint, data=None, params=None, stream=False) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """发送HTTP请求到API服务器"""
        url = f"{self.base_url}{self.api_endpoints.get(endpoint, endpoint)}"
        
        if stream is False:
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=data if method.lower() in ["post", "put", "patch"] else None,
                    params=params if method.lower() == "get" else None,
                    timeout=self.timeout
                )
                
                # 检查响应状态码
                if response.status_code >= 400:
                    error_msg = f"API请求失败: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "status": "failed",
                        "message": error_msg,
                        "code": response.status_code
                    }
                # 解析JSON响应
                result = response.json()
                return result

            except requests.RequestException as e:
                error_msg = f"API请求异常: {str(e)}"
                logger.error(error_msg)
                return {"status": "failed", "message": error_msg}
            except json.JSONDecodeError as e:
                error_msg = f"JSON解析错误: {str(e)}"
                logger.error(error_msg)
                return {"status": "failed", "message": error_msg}
            except Exception as e:
                error_msg = f"未知错误: {str(e)}"
                logger.error(error_msg)
                return {"status": "failed", "message": error_msg}
        else:
            try:
                with requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=data if method.lower() in ["post", "put", "patch"] else None,
                    params=params if method.lower() == "get" else None,
                    stream=True,
                    timeout=self.timeout
                ) as response:
                    # 检查响应状态码
                    if response.status_code >= 400:
                        error_msg = f"API请求失败: {response.status_code} - {response.text}"
                        logger.error(error_msg)
                        yield {
                            "status": "failed",
                            "message": error_msg,
                            "done": True,
                            "code": response.status_code
                        }
                        return
                    # 处理流式响应数据
                    for chunk_data in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk_data:
                            # 解析JSON数据
                            chunk = json.loads(chunk_data)
                            yield {
                                "status": "success",
                                "message": chunk.get("content", ""),
                                "done": chunk.get("done", False),
                                "code": response.status_code
                            }

            except requests.RequestException as e:
                error_msg = f"API请求异常: {str(e)}"
                logger.error(error_msg)
                yield {"status": "failed", "message": error_msg}
            except json.JSONDecodeError as e:
                error_msg = f"JSON解析错误: {str(e)}"
                logger.error(error_msg)
                yield {"status": "failed", "message": error_msg}
            except Exception as e:
                error_msg = f"未知错误: {str(e)}"
                logger.error(error_msg)
                yield {"status": "failed", "message": error_msg}


    
    def inference(
        self, 
        conversation: list, 
        model_name_or_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        device_map: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        documents: Optional[str] = None,
        tools: Optional[list] = None,
        feedback: Optional[str] = None,
        source: Optional[str] = None,
        score: Optional[float] = None,
        notes: Optional[str] = None,
        **kwargs: Optional[Dict[str, Any]]  # 其他参数
    ) -> Iterator[Dict[str, Any]]:
        """发送推理请求到API服务器"""
        # 构建请求数据
        request_data = {
            "conversation": conversation,
        }
        
        # 添加可选参数（如果有值）
        if model_name_or_path is not None:
            request_data["model_name_or_path"] = model_name_or_path
        else:
            request_data["model_name_or_path"] = MODEL_CONFIG.get("model_name_or_path", "SmallDoge/Doge-20M-Instruct")
        if trust_remote_code is not None:
            request_data["trust_remote_code"] = trust_remote_code
        if device_map is not None:
            request_data["device_map"] = device_map
        if torch_dtype is not None:
            request_data["torch_dtype"] = torch_dtype
        if documents is not None:
            request_data["documents"] = documents
        if tools is not None:
            request_data["tools"] = tools
        if feedback is not None:
            request_data["feedback"] = feedback
        if source is not None:
            request_data["source"] = source
        if score is not None:
            request_data["score"] = score
        if notes is not None:
            request_data["notes"] = notes
        
        # 发送请求并获取流式响应
        logger.info(f"向模型发送推理请求: {request_data}")
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 使用流式请求处理
            for chunk in self.make_request("POST", "inference", data=request_data, stream=True):
                yield chunk
                
                # 如果完成则记录耗时
                if chunk.get("done", False):
                    end_time = time.time() - start_time
                    logger.info(f"模型推理完成，耗时: {end_time:.2f}秒")
                    
        except Exception as e:
            error_msg = f"推理请求执行异常: {str(e)}"
            logger.error(error_msg)
            yield {"status": "error", "message": error_msg, "content": "", "done": True}

        

    
    def set_public_status(self, question_id: int, is_public: bool) -> Dict[str, Any]:
        """设置问题公开状态"""
        # 检查登录状态
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}/public_info/set"
        data = {
            "question_id": question_id,
            "is_public": is_public
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = "设置公开状态失败"
                try:
                    error_detail = response.json().get("detail", "未知错误")
                    error_msg = f"设置公开状态失败: {error_detail}"
                except Exception:
                    pass
                return {"status": "error", "message": error_msg}
        except requests.exceptions.RequestException as e:
            logger.error(f"设置公开状态请求失败: {str(e)}")
            return {"status": "error", "message": f"设置公开状态请求失败: {str(e)}"}
    
    # 以下是管理员功能
    def list_public_info(self, status: str = "all", page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """获取公开信息列表（管理员）"""
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}{self.api_endpoints.get('public_info_list')}?status={status}&page={page}&page_size={page_size}"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": "获取公开信息列表失败"}
        except requests.exceptions.RequestException as e:
            logger.error(f"获取公开信息列表失败: {str(e)}")
            return {"status": "error", "message": f"获取公开信息列表失败: {str(e)}"}
    
    def update_public_info_status(self, public_info_id: int, status: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """更新公开信息状态（管理员）"""
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}{self.api_endpoints.get('public_info_update').format(public_info_id=public_info_id)}"
        data = {
            "status": status,
            "notes": notes
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        try:
            response = requests.put(url, json=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": "更新公开信息状态失败"}
        except requests.exceptions.RequestException as e:
            logger.error(f"更新公开信息状态失败: {str(e)}")
            return {"status": "error", "message": f"更新公开信息状态失败: {str(e)}"}
    
    def add_knowledge_base(self, question_content, answer_content, source_id=None, relevance_score=0.5):
        """添加知识库条目（管理员）"""
        # 检查登录状态
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}/api/knowledge_base/add"
        data = {
            "question_content": question_content,
            "answer_content": answer_content,
            "source_id": source_id,
            "relevance_score": relevance_score
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": "添加知识库条目失败"}
        except requests.exceptions.RequestException as e:
            logger.error(f"添加知识库条目失败: {str(e)}")
            return {"status": "error", "message": f"添加知识库条目失败: {str(e)}"}
    
    def add_from_public_info(self, public_info_id, relevance_score=0.5, add_to_training=True, label="medical_qa"):
        """从公开信息添加到知识库和训练集（管理员）"""
        # 检查登录状态
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}/api/knowledge_base/add_from_public_info"
        data = {
            "public_info_id": public_info_id,
            "relevance_score": relevance_score,
            "add_to_training_data": add_to_training,
            "label": label
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = "从公开信息添加到知识库失败"
                try:
                    error_detail = response.json().get("detail", "未知错误")
                    error_msg = f"操作失败: {error_detail}"
                except:
                    pass
                return {"status": "error", "message": error_msg}
        except requests.exceptions.RequestException as e:
            logger.error(f"从公开信息添加到知识库失败: {str(e)}")
            return {"status": "error", "message": f"从公开信息添加到知识库失败: {str(e)}"}

    def search_knowledge_base(self, query, top_k=5):
        """搜索知识库"""
        # 检查登录状态
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}/api/knowledge_base/search?query={query}&top_k={top_k}"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": "搜索知识库失败"}
        except requests.exceptions.RequestException as e:
            logger.error(f"搜索知识库失败: {str(e)}")
            return {"status": "error", "message": f"搜索知识库失败: {str(e)}"}
    
    def start_model_training(
        self, dataset_id=None, learning_rate=5e-5, batch_size=4, epochs=3, 
        label_filter=None, quality_threshold=0.6, **advanced_params
    ):
        """启动模型训练（管理员）"""
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}/api/model/train"
        
        # 构建配置
        config = {
            "dataset_id": dataset_id,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "label_filter": label_filter,
            "quality_threshold": quality_threshold,
        }
        
        # 添加高级参数
        for key, value in advanced_params.items():
            config[key] = value
        
        data = {"config": config}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = "启动模型训练失败"
                try:
                    error_detail = response.json().get("detail", "未知错误")
                    error_msg = f"启动失败: {error_detail}"
                except:
                    pass
                return {"status": "error", "message": error_msg}
        except requests.exceptions.RequestException as e:
            logger.error(f"启动模型训练失败: {str(e)}")
            return {"status": "error", "message": f"启动模型训练失败: {str(e)}"}
    
    def load_trained_model(self, task_id):
        """加载训练完成的模型"""
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}/api/model/load/{task_id}"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = "加载模型失败"
                try:
                    error_detail = response.json().get("detail", "未知错误")
                    error_msg = f"加载失败: {error_detail}"
                except:
                    pass
                return {"status": "error", "message": error_msg}
        except requests.exceptions.RequestException as e:
            logger.error(f"加载模型失败: {str(e)}")
            return {"status": "error", "message": f"加载模型失败: {str(e)}"}
    
    def get_training_status(self, task_id):
        """获取训练状态"""
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}/api/model/status/{task_id}"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": "获取训练状态失败"}
        except requests.exceptions.RequestException as e:
            logger.error(f"获取训练状态失败: {str(e)}")
            return {"status": "error", "message": f"获取训练状态失败: {str(e)}"}
    
    def stop_training(self, task_id):
        """停止模型训练"""
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}/api/model/stop"
        data = {"task_id": task_id}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": "停止训练失败"}
        except requests.exceptions.RequestException as e:
            logger.error(f"停止训练失败: {str(e)}")
            return {"status": "error", "message": f"停止训练失败: {str(e)}"}
    
    def list_training_tasks(self):
        """列出所有训练任务"""
        login_status = self.load_login_state()
        if login_status.get("status") != "success":
            return {"status": "error", "message": "未登录"}
        
        url = f"{self.base_url}/api/model/list"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": "获取训练任务列表失败"}
        except requests.exceptions.RequestException as e:
            logger.error(f"获取训练任务列表失败: {str(e)}")
            return {"status": "error", "message": f"获取训练任务列表失败: {str(e)}"}