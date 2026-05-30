# config.py  项目配置，改这里就行不用动app.py
# 数据库和flask的参数都在这

# ---- 数据库配置 ----
# 我这里装的是XAMPP，root密码是root，如果你用workbench记得改
# 端口默认3306不用动
DB_HOST     = "localhost"
DB_PORT     = 3306
DB_USER     = "root"
DB_PASSWORD = "root"   # XAMPP默认是空字符串""，我改过密码所以是root
DB_NAME     = "BISHE"  # 数据库不用自己建，运行check_db.py会自动创建

# ---- 模型和文件路径 ----
# 注意：这里用的是相对路径，要保证项目结构和ultralytics001在同一层

FILE_EXCEL_PATH = "./ultralytics001/yolo_obb/序号标记对照表.xlsx"
MODEL_PATH1 = "./ultralytics001/yolo_obb/weight/1biaopan_all/weights/best.pt"
MODEL_PATH2 = "./ultralytics001/yolo_obb/weight/2biaopan_nolabel/weights/best.pt"
MODEL_PATH3 = "./ultralytics001/yolo_obb/weight/3biaopan_label/weights/best.pt"
MODEL_PATH4 = "./ultralytics001/yolo_obb/weight/4read/weights/best.pt"
TXT_LOG_PATH = "./ultralytics001/yolo_obb/Result_pointer.txt"

# 上传和输出文件夹，会自动创建
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# ---- Flask配置 ----
FLASK_SECRET_KEY = "bishe_flask_secret_2025"  # 随便写的，上线记得换
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True  # 调试模式，部署的时候改False

# ---- 二维码配置 ----
# 微信小程序路径（格式：小程序原始ID或小程序链接）
# 例如：weixin://dl/business/?t=xxx 或者小程序scheme
WECHAT_MINIPROGRAM_PATH = "pages/index/index"  # 小程序页面路径
WECHAT_MINIPROGRAM_APPID = "your_miniprogram_appid"  # 小程序AppID，替换成实际的

# APK下载链接（可以是服务器上的静态文件路径或CDN链接）
APK_DOWNLOAD_URL = "https://your-domain.com/downloads/app.apk"  # 替换成实际下载链接
# 或者使用本地路径："/static/downloads/app.apk"
