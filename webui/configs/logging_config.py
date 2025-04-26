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

# 日志配置 (Logging Configuration)
LOGGING_CONFIG = {
    # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL (Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL)
    'level': 'INFO',
    
    # 日志格式，包含时间、名称、级别、文件路径、行号和消息
    # Log format including time, name, level, file path, line number and message
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
    
    # 日志文件路径，None表示不写入文件仅控制台输出 (Log file path, None means console output only)
    'file': 'logs/webui.log',
    
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