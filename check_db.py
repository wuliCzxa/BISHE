"""
check_db.py
数据库连接检测 + 自动建表脚本
先运行这个确认数据库没问题再跑app.py

用法：python check_db.py
"""

import sys
import socket

try:
    import config as cfg
except ImportError:
    print("找不到config.py，把两个文件放同一个文件夹里")
    sys.exit(1)

try:
    import pymysql
    import pymysql.cursors
except ImportError:
    print("没装pymysql，先执行：pip install pymysql")
    sys.exit(1)

try:
    from werkzeug.security import generate_password_hash
except ImportError:
    print("没装flask，先执行：pip install flask")
    sys.exit(1)


print("\n========== 检测MySQL端口是否可达 ==========")
host = cfg.DB_HOST
port = cfg.DB_PORT
try:
    s = socket.create_connection((host, port), timeout=3)
    s.close()
    print(f"OK  {host}:{port} 端口通了，MySQL应该在跑")
except Exception as e:
    print(f"连不上 {host}:{port}，错误：{e}")
    print("""
MySQL服务没启动，根据你的环境选：

XAMPP用户：打开XAMPP Control Panel，点MySQL右边的Start
MySQL Installer：Win+R -> services.msc -> 找MySQL80 -> 右键启动
命令行(管理员)：net start MySQL80
Linux：sudo systemctl start mysql

启动之后重新跑这个脚本
""")
    sys.exit(1)


print("\n========== 验证用户名密码 ==========")
conn_cfg = {
    "host":     cfg.DB_HOST,
    "port":     cfg.DB_PORT,
    "user":     cfg.DB_USER,
    "password": cfg.DB_PASSWORD,
    "charset":  "utf8mb4",
    "connect_timeout": 5,
}

conn_server = None
try:
    conn_server = pymysql.connect(**conn_cfg, cursorclass=pymysql.cursors.DictCursor)
    print(f"登录成功，用户：{cfg.DB_USER}")
except pymysql.err.OperationalError as e:
    code, msg = e.args
    print(f"登录失败（错误码{code}）：{msg}")
    if code == 1045:
        print(f"""
密码错误！
去config.py把DB_PASSWORD改成正确的密码
XAMPP默认密码是空字符串：DB_PASSWORD = ""
""")
    sys.exit(1)


print("\n========== 创建BISHE数据库 ==========")
try:
    with conn_server.cursor() as cur:
        # utf8mb4支持emoji，反正加上没坏处
        cur.execute(
            "CREATE DATABASE IF NOT EXISTS `BISHE` "
            "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    conn_server.commit()
    print("BISHE数据库已就绪")
except Exception as e:
    print(f"建库失败：{e}")
    sys.exit(1)
finally:
    conn_server.close()


print("\n========== 创建数据表 ==========")
# 重新连接，这次指定数据库
full_cfg = {**conn_cfg, "database": cfg.DB_NAME, "autocommit": False}

try:
    conn = pymysql.connect(**full_cfg, cursorclass=pymysql.cursors.DictCursor)
except Exception as e:
    print(f"连接BISHE数据库失败：{e}")
    sys.exit(1)

try:
    with conn.cursor() as cur:
        # user表，存账号密码
        cur.execute("""
        CREATE TABLE IF NOT EXISTS `user` (
            `id`         INT UNSIGNED NOT NULL AUTO_INCREMENT,
            `username`   VARCHAR(64)  NOT NULL,
            `password`   VARCHAR(255) NOT NULL,
            `user_level` ENUM('super_admin','admin','user') NOT NULL DEFAULT 'user',
            `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(`id`),
            UNIQUE KEY `uq_username`(`username`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("user表 OK")

        # yolo表，存每次检测任务的结果
        # reading_before是修正前，reading_after是修正后（用户可以手动改）
        cur.execute("""
        CREATE TABLE IF NOT EXISTS `yolo` (
            `id`                INT UNSIGNED  NOT NULL AUTO_INCREMENT,
            `task_id`           VARCHAR(32)   NOT NULL,
            `user_id`           INT UNSIGNED  NOT NULL,
            `serial_number`     VARCHAR(64)       NULL DEFAULT NULL,
            `original_img_path` VARCHAR(512)  NOT NULL,
            `dial_img_path`     VARCHAR(512)      NULL DEFAULT NULL,
            `label_img_path`    VARCHAR(512)      NULL DEFAULT NULL,
            `obb_img_path`      VARCHAR(512)      NULL DEFAULT NULL,
            `reading_before`    DECIMAL(12,6)     NULL DEFAULT NULL,
            `reading_after`     DECIMAL(12,6)     NULL DEFAULT NULL,
            `detect_status`     ENUM('pending','running','success','failed') NOT NULL DEFAULT 'pending',
            `is_confirmed`      TINYINT(1)    NOT NULL DEFAULT 0,
            `confirmed_at`      DATETIME          NULL DEFAULT NULL,
            `detected_at`       DATETIME          NULL DEFAULT NULL,
            `created_at`        DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(`id`),
            UNIQUE KEY `uq_task_id`(`task_id`),
            KEY `idx_user_id`(`user_id`),
            KEY `idx_status`(`detect_status`),
            CONSTRAINT `fk_yolo_user`
                FOREIGN KEY(`user_id`) REFERENCES `user`(`id`)
                ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("yolo表 OK")

        # # 默认管理员账号，第一次跑的时候写进去
        # cur.execute("SELECT id FROM `user` WHERE username='admin' LIMIT 1")
        # if not cur.fetchone():
        #     pw = generate_password_hash("admin123")
        #     cur.execute(
        #         "INSERT INTO `user`(username,password,user_level) VALUES(%s,%s,%s)",
        #         ("admin", pw, "super_admin")
        #     )
        #     print("已创建默认账号：admin / admin123")
        # else:
        #     print("admin账号已存在，跳过")

    conn.commit()
    print("建表完成")

except Exception as e:
    print(f"建表出错：{e}")
    # conn.rollback()  # 出错要回滚，但这里DDL语句不需要
    sys.exit(1)
finally:
    conn.close()


print(f"""
========== 全部通过！==========

数据库：{cfg.DB_HOST}:{cfg.DB_PORT} / {cfg.DB_NAME}
用户：{cfg.DB_USER}
admin账号：admin / admin123
super账号：super / 123456

现在可以运行：
    python app.py

然后浏览器打开：
    http://localhost:{cfg.FLASK_PORT}/login
""")
