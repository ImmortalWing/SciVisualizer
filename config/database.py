import sqlite3
from contextlib import contextmanager

DATABASE_PATH = "data/visualization.db"

@contextmanager
def get_connection():
    """数据库连接上下文管理器"""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()

def execute_query(query: str, parameters=()):
    """执行查询并返回游标对象"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, parameters)
        conn.commit()
        return cursor  # 返回游标以便访问description属性