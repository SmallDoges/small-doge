import os

# 数据库配置 (Database Configuration)
# 此配置用于连接到MySQL/MariaDB数据库
# This configuration is used for connecting to MySQL/MariaDB database

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),  # 数据库主机地址 (Database host address)
    'port': int(os.environ.get('DB_PORT', 3306)),    # 数据库端口 (Database port)
    'user': os.environ.get('DB_USER', 'root'),       # 数据库用户名 (Database username)
    'password': os.environ.get('DB_PASSWORD', 'yjzbzdsbknd123'),  # 数据库密码 (Database password)
    'database': os.environ.get('DB_NAME', 'medical_qa_system'),   # 数据库名称 (Database name)
    'charset': 'utf8mb4',                            # 字符集编码 (Character encoding)
    'cursorclass': 'DictCursor',                     # 游标类型，返回字典格式结果 (Cursor type, returns results as dictionaries)
    'autocommit': True                               # 自动提交事务 (Auto-commit transactions)
}