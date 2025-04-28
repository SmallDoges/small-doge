import pymysql
from pymysql.cursors import DictCursor
from datetime import datetime
import logging
from ..configs.logging_config import LOGGING_CONFIG
from ..configs.db_config import DB_CONFIG

# 配置日志记录
logging.basicConfig(**LOGGING_CONFIG)
# 设置日志记录器
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """数据库连接管理类"""
    def __init__(self, host=None, port=None, user=None, password=None, database=None):
        self.host = host or DB_CONFIG['host']
        self.port = port or DB_CONFIG['port']
        self.user = user or DB_CONFIG['user']
        self.password = password or DB_CONFIG['password']
        self.database = database or DB_CONFIG['database']
        self.connection = None
    
    def connect(self):
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                cursorclass=DictCursor
            )
            return True
        except pymysql.MySQLError as e:
            logger.error(f"数据库连接错误: {e}")
            return False
    
    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def execute_query(self, query, params=None):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or ())
                self.connection.commit()
                return cursor
        except pymysql.MySQLError as e:
            self.connection.rollback()
            logger.error(f"查询执行错误: {e}")
            logger.error(f"SQL: {query}")
            logger.error(f"参数: {params}")
            raise


class DatabaseInitializer:
    """数据库表初始化类"""
    def __init__(self, db_connection):
        self.db_connection = db_connection
    
    def create_tables(self):
        """创建所有数据表"""
        if not self.db_connection.connect():
            logger.error("无法连接到数据库")
            return False
        
        try:
            self._create_message_table()
            return True
        except Exception as e:
            logger.error(f"创建表时出错: {e}")
            return False
        finally:
            self.db_connection.disconnect()
    
    def _create_message_table(self):
        """创建消息表"""
        query = """
        CREATE TABLE IF NOT EXISTS Messages (
            MessageID INTEGER PRIMARY KEY AUTO_INCREMENT,
            Message JSON NOT NULL,
            Timestamp DATETIME NOT NULL,
            Model_Path_or_Name VARCHAR(100) NULL,
            Feedback TEXT NULL,
            Source VARCHAR(100) NULL,
            Score FLOAT DEFAULT 0.0,
            Notes TEXT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        self.db_connection.execute_query(query)
        logger.info("消息表创建成功")


class MessagesModel:
    """消息表CRUD操作"""
    def __init__(self, db_connection):
        self.db_connection = db_connection
    
    def create(self, message_json, model_path_or_name=None, feedback=None, source=None, score=0.0, notes=None):
        """创建新消息"""
        query = """
        INSERT INTO Messages 
        (Message, Timestamp, Model_Path_or_Name, Feedback, Source, Score, Notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        current_time = datetime.now()
        params = (message_json, current_time, model_path_or_name, feedback, source, score, notes)
        
        if not self.db_connection.connect():
            return False
        
        try:
            cursor = self.db_connection.execute_query(query, params)
            message_id = cursor.lastrowid
            return message_id
        except Exception as e:
            logger.error(f"创建消息失败: {e}")
            return False
        finally:
            self.db_connection.disconnect()
    
    def get_by_id(self, message_id):
        """根据ID获取消息"""
        query = "SELECT * FROM Messages WHERE MessageID = %s"
        
        if not self.db_connection.connect():
            return None
        
        try:
            cursor = self.db_connection.execute_query(query, (message_id,))
            return cursor.fetchone()
        except:
            return None
        finally:
            self.db_connection.disconnect()
    
    def update(self, message_id, **kwargs):
        """更新消息"""
        allowed_fields = ['Message', 'Model_Path_or_Name', 'Feedback', 'Source', 'Score', 'Notes']
        
        field_updates = []
        params = []
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                field_updates.append(f"{key} = %s")
                params.append(value)
        
        if not field_updates:
            return False
        
        query = f"UPDATE Messages SET {', '.join(field_updates)} WHERE MessageID = %s"
        params.append(message_id)
        
        if not self.db_connection.connect():
            return False
        
        try:
            self.db_connection.execute_query(query, tuple(params))
            return True
        except:
            return False
        finally:
            self.db_connection.disconnect()
    
    def delete(self, message_id):
        """删除消息"""
        query = "DELETE FROM Messages WHERE MessageID = %s"
        
        if not self.db_connection.connect():
            return False
        
        try:
            self.db_connection.execute_query(query, (message_id,))
            return True
        except:
            return False
        finally:
            self.db_connection.disconnect()
    
    def get_all(self, limit=100, offset=0):
        """获取所有消息"""
        query = "SELECT * FROM Messages ORDER BY Timestamp DESC LIMIT %s OFFSET %s"
        
        if not self.db_connection.connect():
            return []
        
        try:
            cursor = self.db_connection.execute_query(query, (limit, offset))
            return cursor.fetchall()
        except:
            return []
        finally:
            self.db_connection.disconnect()
    
    def update_feedback(self, message_id, feedback):
        """更新消息反馈"""
        query = "UPDATE Messages SET Feedback = %s WHERE MessageID = %s"
        
        if not self.db_connection.connect():
            return False
        
        try:
            self.db_connection.execute_query(query, (feedback, message_id))
            return True
        except:
            return False
        finally:
            self.db_connection.disconnect()
    
    def search_by_source(self, source, limit=50, offset=0):
        """根据来源搜索消息"""
        query = "SELECT * FROM Messages WHERE Source LIKE %s ORDER BY Timestamp DESC LIMIT %s OFFSET %s"
        search_param = f"%{source}%"
        
        if not self.db_connection.connect():
            return []
        
        try:
            cursor = self.db_connection.execute_query(query, (search_param, limit, offset))
            return cursor.fetchall()
        except:
            return []
        finally:
            self.db_connection.disconnect()
    
    def search_by_score(self, min_score, max_score, limit=50, offset=0):
        """根据评分范围搜索消息"""
        query = "SELECT * FROM Messages WHERE Score BETWEEN %s AND %s ORDER BY Timestamp DESC LIMIT %s OFFSET %s"
        
        if not self.db_connection.connect():
            return []
        
        try:
            cursor = self.db_connection.execute_query(query, (min_score, max_score, limit, offset))
            return cursor.fetchall()
        except:
            return []
        finally:
            self.db_connection.disconnect()