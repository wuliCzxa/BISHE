"""
check_db.py — MySQL 连接诊断 & 自动修复脚本
=============================================
运行方式：
    python check_db.py

功能：
  1. 检测 MySQL 服务是否可达
  2. 验证用户名/密码
  3. 自动创建 BISHE 数据库和数据表
  4. 写入默认管理员 admin / admin123
  5. 输出详细的修复建议
"""

import sys
import socket
import datetime

# ── 从 config.py 读取配置 ──────────────────────────────────
try:
    import config as cfg
except ImportError:
    print("❌ 找不到 config.py，请确保 check_db.py 和 config.py 在同一目录")
    sys.exit(1)

# ── 检查 pymysql ───────────────────────────────────────────
try:
    import pymysql
    import pymysql.cursors
except ImportError:
    print("❌ 未安装 pymysql，请先执行：pip install pymysql")
    sys.exit(1)

# ── 检查 werkzeug ──────────────────────────────────────────
try:
    from werkzeug.security import generate_password_hash
except ImportError:
    print("❌ 未安装 flask/werkzeug，请先执行：pip install flask")
    sys.exit(1)

SEP = "─" * 60

def banner(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ══════════════════════════════════════════════════════════
#  Step 1：网络层检测（TCP 端口是否可达）
# ══════════════════════════════════════════════════════════
banner("Step 1 / 4 · 检测 MySQL 端口是否可达")

host, port = cfg.DB_HOST, cfg.DB_PORT
try:
    sock = socket.create_connection((host, port), timeout=3)
    sock.close()
    print(f"  ✔ {host}:{port} 端口可达，MySQL 服务正在运行")
    port_ok = True
except (socket.timeout, ConnectionRefusedError, OSError) as e:
    print(f"  ❌ 无法连接到 {host}:{port} — {e}")
    port_ok = False
    print("""
  【解决方法】MySQL 服务未启动，请根据你的环境选择：

  ▸ Windows + XAMPP：
      打开 XAMPP Control Panel → 点击 MySQL 的 [Start] 按钮

  ▸ Windows + MySQL Installer：
      按 Win+R → 输入 services.msc → 找到 "MySQL80" 或 "MySQL" → 右键启动

  ▸ Windows 命令行（管理员）：
      net start MySQL80

  ▸ Linux / macOS：
      sudo systemctl start mysql
      或
      sudo service mysql start

  ▸ macOS (Homebrew)：
      brew services start mysql

  启动后重新运行此脚本。
""")
    sys.exit(1)

# ══════════════════════════════════════════════════════════
#  Step 2：用户名 / 密码 验证
# ══════════════════════════════════════════════════════════
banner("Step 2 / 4 · 验证用户名和密码")

cfg_no_db = {
    "host":    cfg.DB_HOST,
    "port":    cfg.DB_PORT,
    "user":    cfg.DB_USER,
    "password": cfg.DB_PASSWORD,
    "charset": "utf8mb4",
    "connect_timeout": 5,
}

conn_server = None
try:
    conn_server = pymysql.connect(**cfg_no_db, cursorclass=pymysql.cursors.DictCursor)
    print(f"  ✔ 用户 '{cfg.DB_USER}' 登录成功")
except pymysql.err.OperationalError as e:
    code, msg = e.args
    print(f"  ❌ 登录失败（错误 {code}）：{msg}")
    if code == 1045:
        print(f"""
  【解决方法】密码错误，请：
    1. 打开 config.py
    2. 修改 DB_PASSWORD = "正确的密码"
    3. 重新运行此脚本

  ▸ 如果你用 XAMPP，root 默认密码通常是空字符串：DB_PASSWORD = ""
  ▸ 忘记密码？MySQL 重置密码教程：
    https://dev.mysql.com/doc/mysql-windows-excerpt/8.0/en/resetting-permissions-windows.html
""")
    elif code == 2003:
        print("  MySQL 服务端口可达但拒绝连接，请检查 DB_USER 是否正确")
    sys.exit(1)

# ══════════════════════════════════════════════════════════
#  Step 3：创建数据库 BISHE
# ══════════════════════════════════════════════════════════
banner("Step 3 / 4 · 创建/确认数据库 BISHE")

try:
    with conn_server.cursor() as cur:
        cur.execute(
            "CREATE DATABASE IF NOT EXISTS `BISHE` "
            "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    conn_server.commit()
    print(f"  ✔ 数据库 BISHE 已就绪")
except Exception as e:
    print(f"  ❌ 创建数据库失败：{e}")
    sys.exit(1)
finally:
    conn_server.close()

# ══════════════════════════════════════════════════════════
#  Step 4：创建数据表 & 默认账号
# ══════════════════════════════════════════════════════════
banner("Step 4 / 4 · 创建数据表 & 默认账号")

cfg_with_db = {**cfg_no_db, "database": cfg.DB_NAME, "autocommit": False}

try:
    conn = pymysql.connect(**cfg_with_db, cursorclass=pymysql.cursors.DictCursor)
except Exception as e:
    print(f"  ❌ 连接到 BISHE 数据库失败：{e}")
    sys.exit(1)

try:
    with conn.cursor() as cur:
        # user 表
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
        print("  ✔ user 表已就绪")

        # yolo 表
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
            `detect_status`     ENUM('pending','running','success','failed')
                                              NOT NULL DEFAULT 'pending',
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
        print("  ✔ yolo 表已就绪")

        # # 默认管理员
        # cur.execute("SELECT id FROM `user` WHERE username='admin' LIMIT 1")
        # if not cur.fetchone():
        #     pw_hash = generate_password_hash("admin123")
        #     cur.execute(
        #         "INSERT INTO `user`(username,password,user_level) VALUES(%s,%s,%s)",
        #         ("admin", pw_hash, "super_admin")
        #     )
        #     print("  ✔ 默认账号已创建：admin / admin123")
        # else:
        #     print("  ✔ 默认账号 admin 已存在")

    conn.commit()
    print("  ✔ 所有数据表初始化完成")

except Exception as e:
    print(f"  ❌ 建表失败：{e}")
    sys.exit(1)
finally:
    conn.close()

# ══════════════════════════════════════════════════════════
#  汇总结果
# ══════════════════════════════════════════════════════════
banner("✅  全部检测通过！")
print(f"""
  数据库地址 ：{cfg.DB_HOST}:{cfg.DB_PORT}
  数据库名称 ：{cfg.DB_NAME}
  登录用户   ：{cfg.DB_USER}

  # 默认账号   ：admin
  # 默认密码   ：admin123

  现在可以运行：
      python app.py

  然后浏览器访问：
      http://localhost:{cfg.FLASK_PORT}/login
""")
