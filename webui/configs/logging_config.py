# coding=utf-8
# Copyright 2025 SmallDoge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
from logging.handlers import RotatingFileHandler

# 确保日志目录存在
os.makedirs('logs', exist_ok=True)

# 日志配置 (Logging Configuration)
LOGGING_CONFIG = {
    # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL (Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL)
    'level': logging.INFO,
    
    # 日志格式，包含时间、名称、级别、文件路径、行号和消息
    # Log format including time, name, level, file path, line number and message
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
    
    # 日期时间格式 (Date/time format)
    'datefmt': '%Y-%m-%d %H:%M:%S',
    
    # 格式化风格 (Format style: '%', '{', '$')
    'style': '%',
    
    # 日志文件路径，None表示不写入文件仅控制台输出 (Log file path, None means console output only)
    'filename': 'logs/webui.log',
    
    # 日志文件打开模式 (File mode: 'a' for append, 'w' for overwrite)
    'filemode': 'a',
    
    # 文件编码 (File encoding)
    'encoding': 'utf-8',
    
    # 文件打开错误处理模式 (File open error handling mode)
    'errors': 'backslashreplace',
    
    # 是否强制重新配置日志处理器 (Whether to force reconfiguration of log handlers)
    'force': False
}

# 扩展日志配置 (Extended logging configuration)
# 这些配置不是 logging.basicConfig 直接支持的，但在创建自定义处理器时会用到
EXTENDED_LOGGING_CONFIG = {
    # 日志文件最大大小，单位MB (Maximum log file size in MB)
    'max_size_mb': 10,
    
    # 保留的日志文件数量 (Number of log files to keep)
    'backup_count': 5,
    
    # 是否在控制台输出日志 (Whether to output logs to console)
    'console_output': True,
    
    # 异常详细信息 (Exception details)
    'capture_exceptions': True,
    
    # 是否启用请求日志记录 (Whether to enable request logging)
    'log_requests': True
}

def setup_custom_logger(name):
    """
    设置自定义日志记录器，支持同时输出到控制台和文件，并支持日志文件滚动
    Setup a custom logger that outputs to both console and file with log rotation support
    
    Args:
        name: 日志记录器名称 (Logger name)
    
    Returns:
        logger: 配置好的日志记录器 (Configured logger)
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_CONFIG['level'])
    
    formatter = logging.Formatter(
        fmt=LOGGING_CONFIG['format'],
        datefmt=LOGGING_CONFIG['datefmt'],
        style=LOGGING_CONFIG['style']
    )
    
    # 添加文件处理器 (Add file handler)
    if LOGGING_CONFIG['filename']:
        file_handler = RotatingFileHandler(
            LOGGING_CONFIG['filename'],
            mode=LOGGING_CONFIG['filemode'],
            maxBytes=EXTENDED_LOGGING_CONFIG['max_size_mb'] * 1024 * 1024,
            backupCount=EXTENDED_LOGGING_CONFIG['backup_count'],
            encoding=LOGGING_CONFIG['encoding'],
            errors=LOGGING_CONFIG['errors']
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 添加控制台处理器 (Add console handler)
    if EXTENDED_LOGGING_CONFIG['console_output']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger