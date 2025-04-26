import os

# SmallDoges WebUI 配置 (SmallDoges WebUI Configuration)
APP_CONFIG = {
    'debug': True,                 # 调试模式 (Debug mode)
    'host': '0.0.0.0',             # 服务器主机地址 (Server host address)
    'port': 7860,                  # 服务器端口 (Server port)
    'title': 'SmallDoges',         # 应用标题 (Application title)
    'description': 'All-in-one WebUI for Inference and Training',  # 应用描述 (Application description)
    'share': True,                 # 是否共享应用 (Whether to share the application)
    'auth_enabled': True,          # 是否启用认证 (Whether to enable authentication)
    'theme': 'default',            # UI主题 (UI theme)
    'assets_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets'),  # 资源目录路径 (Assets directory path)
    'favicon': 'favicon.ico',      # 网站图标 (Website icon)
    'allowed_paths': ['assets'],   # 允许访问的路径 (Allowed access paths)
    'quiet': True,                 # 静默模式 (Quiet mode)
    'show_api': False,             # 是否显示API (Whether to show API)
    'root_path': "",               # 根路径 (Root path)
    'ssl_verify': False,           # SSL验证 (SSL verification)
    'prevent_thread_lock': True,   # 防止线程锁 (Prevent thread lock)
    'max_connections': 100,        # 最大连接数 (Maximum number of connections)
    'timeout': 30,                 # 超时时间(秒) (Timeout duration in seconds)
    'enable_logging': False        # 启用日志记录 (Enable logging)
}