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

# Ensure log directory exists
os.makedirs('logs', exist_ok=True)

# Logging Configuration
LOGGING_CONFIG = {
    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    'level': logging.INFO,
    
    # Log format including time, name, level, file path, line number and message
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
    
    # Date/time format
    'datefmt': '%Y-%m-%d %H:%M:%S',
    
    # Format style: '%', '{', '$'
    'style': '%',
    
    # Log file path, None means console output only
    'filename': 'logs/webui.log',
    
    # File mode: 'a' for append, 'w' for overwrite
    'filemode': 'a',
    
    # File encoding
    'encoding': 'utf-8',
    
    # File open error handling mode
    'errors': 'backslashreplace',
    
    # Whether to force reconfiguration of log handlers
    'force': False
}

# Extended logging configuration
# These configurations are not directly supported by logging.basicConfig, but are used when creating custom handlers
EXTENDED_LOGGING_CONFIG = {
    # Maximum log file size in MB
    'max_size_mb': 10,
    
    # Number of log files to keep
    'backup_count': 5,
    
    # Whether to output logs to console
    'console_output': True,
    
    # Exception details
    'capture_exceptions': True,
    
    # Whether to enable request logging
    'log_requests': True
}

def setup_custom_logger(name):
    """
    Setup a custom logger that outputs to both console and file with log rotation support
    
    Args:
        name: Logger name
    
    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_CONFIG['level'])
    
    formatter = logging.Formatter(
        fmt=LOGGING_CONFIG['format'],
        datefmt=LOGGING_CONFIG['datefmt'],
        style=LOGGING_CONFIG['style']
    )
      # Add file handler
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
    
    # Add console handler
    if EXTENDED_LOGGING_CONFIG['console_output']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger