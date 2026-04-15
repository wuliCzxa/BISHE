# ============================================================
#  config.py — BISHE 项目统一配置文件
#  修改此文件后重启 app.py 即可生效
# ============================================================

# ── MySQL 数据库连接 ──────────────────────────────────────
# 常见配置说明：
#   XAMPP / phpMyAdmin  → host="localhost", port=3306, user="root", password=""（空字符串）
#   MySQL Workbench     → host="localhost", port=3306, 填写安装时设置的密码
#   宝塔面板             → host="127.0.0.1", port=3306, user="root", password="宝塔数据库密码"
#   远程服务器           → host="服务器IP", port=3306

DB_HOST     = "localhost"   # 数据库地址
DB_PORT     = 3306          # 端口（默认 3306）
DB_USER     = "root"        # 用户名
DB_PASSWORD = "root"            # ← 填写你的 MySQL 密码（XAMPP 默认为空字符串 ""）
DB_NAME     = "BISHE"       # 数据库名，不需要提前创建，会自动建库

# ── 文件路径 ─────────────────────────────────────────────
FILE_EXCEL_PATH = "ultralytics001/yolo_obb/序号标记对照表.xlsx"          # 序号标记对照表
MODEL_PATH1 = "../ultralytics001/yolo_obb/weight/1biaopan_all/weights/best.pt"
MODEL_PATH2 = "../ultralytics001/yolo_obb/weight/2biaopan_nolabel/weights/best.pt"
MODEL_PATH3 = "../ultralytics001/yolo_obb/weight/3biaopan_label/weights/best.pt"
MODEL_PATH4 = "../ultralytics001/yolo_obb/weight/4read/weights/best.pt"
TXT_LOG_PATH = "ultralytics001/yolo_obb/Result_pointer.txt"

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# ── Flask ────────────────────────────────────────────────
FLASK_SECRET_KEY = "bishe_flask_secret_2025"
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
