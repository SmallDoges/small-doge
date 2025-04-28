# SmallDoges系统开发文档

## 1. 项目概述

### 1.1 开发目标
- 开发一个基于Gradio的用户友好界面
- 构建知识库管理系统和模型训练框架
- 确保数据安全和系统可靠性

### 1.2 技术栈
- 前端：Gradio
- 后端：FastAPI
- 数据库：MySQL
- 机器学习：PyTorch, Hugging Face Transformers
- 知识库：Langchain

## 2. 系统架构

### 2.1 系统层次结构
- **表示层**：Gradio界面
- **应用层**：FastAPI服务
- **数据访问层**：MySQL数据库接口
- **基础设施层**：服务器和存储系统

### 2.2 核心模块
- **用户管理模块**：处理用户注册、登录和权限管理
- **问答模块**：处理用户问题提交和模型回答
- **知识库管理模块**：维护和更新知识库内容
- **模型训练模块**：训练和优化预测模型
- **数据管理模块**：处理数据存储和备份

## 3. 数据库设计

### 3.1 数据库表结构

#### 问题表 (Question)

#### 消息表 (Messages)
| 字段名 | 数据类型 | 约束 | 说明 |
|--------|----------|------|------|
| MessageID | INTEGER | PRIMARY KEY, AUTO_INCREMENT | 消息唯一标识 |
| Message | JSON | NOT NULL | 消息内容，JSON格式 |
| Timestamp | DATETIME | NOT NULL | 消息时间戳 |
| Model_Path_or_Name | VARCHAR(100) | NULL | 模型路径或名称 |
| Feedback | TEXT | NULL | 用户反馈 |
| Source | VARCHAR(100) | NULL | 消息来源 |
| Score | FLOAT | DEFAULT 0.0 | 数据质量评分，范围0-1 |
| Notes | TEXT | NULL | 备注 |

