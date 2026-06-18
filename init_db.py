import os

import mysql.connector

def init_database():
    try:
        # 连接到MySQL服务器
        db = mysql.connector.connect(
            host=os.getenv("SMART_ALBUM_DB_HOST", "localhost"),
            user=os.getenv("SMART_ALBUM_DB_USER", "root"),
            password=os.getenv("SMART_ALBUM_DB_PASSWORD", ""),
        )
        cursor = db.cursor()

        # 创建数据库
        database_name = os.getenv("SMART_ALBUM_DB_NAME", "smart_album")
        if not database_name.replace("_", "").isalnum():
            raise ValueError("SMART_ALBUM_DB_NAME 只能包含字母、数字和下划线")

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database_name}`")
        print('数据库创建成功')

        # 选择数据库
        cursor.execute(f"USE `{database_name}`")

        # 创建照片表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            id INT AUTO_INCREMENT PRIMARY KEY,
            path VARCHAR(255) NOT NULL,
            timestamp DATETIME NOT NULL,
            description TEXT
        )
        ''')
        print('照片表创建成功')

        # 创建人物表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            embedding TEXT
        )
        ''')
        print('人物表创建成功')

        # 创建照片-人物关联表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS photo_person (
            photo_id INT,
            person_id INT,
            FOREIGN KEY (photo_id) REFERENCES photos(id),
            FOREIGN KEY (person_id) REFERENCES persons(id),
            PRIMARY KEY (photo_id, person_id)
        )
        ''')
        print('照片-人物关联表创建成功')

        db.commit()
        cursor.close()
        db.close()
        print('数据库初始化完成')
    except Exception as e:
        print(f"数据库初始化失败: {e}")

if __name__ == '__main__':
    init_database()
