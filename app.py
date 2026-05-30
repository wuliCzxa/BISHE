"""
app.py  指针式仪表读数检测识别系统
基于Flask + MySQL + YOLOv8

安装依赖：
    pip install requirements.txt
    // pip install flask pymysql pandas openpyxl ultralytics opencv-python numpy

运行：
    1. 改config.py里的数据库密码
    2. 先跑python check_db.py确认数据库没问题
    3. python app.py
    4. 浏览器打开 http://localhost:5000/login
       默认账号 admin / admin123
"""
# 顶部需要导入
import ssl

import os, math, datetime, threading, traceback
from math import sqrt
from functools import wraps
from flask_cors import CORS

import jwt
import cv2
import numpy as np
import pandas as pd
import pymysql
import pymysql.cursors
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (Flask, render_template, request, jsonify,
                   send_file, abort, redirect, url_for, session)

import jwt
from datetime import timedelta
# from datetime import timedelta

from ultralytics import YOLO

# 二维码相关功能
from qrcode_helper import serve_qrcode, get_qrcode_info

# 提供静态文件
from flask import send_from_directory

SECRET_KEY = "wuliBISHE_secret_key_2025"  # JWT密钥，生产环境请改成更复杂的值，并保密

# ngrok 相关功能
import webbrowser

NGROK_URL = "https://69c1-2408-862e-807-c000-00-68a.ngrok-free.app/login"

# 微信路由

# 读config.py，找不到就用默认值（一般不会走这个分支）
try:
    import config as _cfg
    _DB_HOST     = _cfg.DB_HOST
    _DB_PORT     = _cfg.DB_PORT
    _DB_USER     = _cfg.DB_USER
    _DB_PASSWORD = _cfg.DB_PASSWORD
    _DB_NAME     = _cfg.DB_NAME
    FILE_EXCEL_PATH = _cfg.FILE_EXCEL_PATH
    MODEL_PATH1  = _cfg.MODEL_PATH1
    MODEL_PATH2  = _cfg.MODEL_PATH2
    MODEL_PATH3  = _cfg.MODEL_PATH3
    MODEL_PATH4  = _cfg.MODEL_PATH4
    TXT_LOG_PATH = _cfg.TXT_LOG_PATH
    UPLOAD_FOLDER= _cfg.UPLOAD_FOLDER
    OUTPUT_FOLDER= _cfg.OUTPUT_FOLDER
    _SECRET_KEY  = _cfg.FLASK_SECRET_KEY
    print("配置加载成功")

    # 添加二维码配置读取
    WECHAT_MINIPROGRAM_PATH = getattr(_cfg, 'WECHAT_MINIPROGRAM_PATH', 'pages/index/index')
    WECHAT_MINIPROGRAM_APPID = getattr(_cfg, 'WECHAT_MINIPROGRAM_APPID', 'your_miniprogram_appid')
    APK_DOWNLOAD_URL = getattr(_cfg, 'APK_DOWNLOAD_URL', 'https://your-domain.com/downloads/app.apk')
    print("二维码配置加载成功")

except (ImportError, AttributeError) as _ce:
    print(f"config.py加载失败({_ce})，用内置默认值")
    _DB_HOST="localhost"; _DB_PORT=3306; _DB_USER="root"; _DB_PASSWORD=""
    _DB_NAME="BISHE"
    FILE_EXCEL_PATH="/ultralytics001/yolo_obb/序号标记对照表.xlsx"
    MODEL_PATH1="/ultralytics001/yolo_obb/weight/1biaopan_all/weights/best.pt"
    MODEL_PATH2="/ultralytics001/yolo_obb/weight/2biaopan_nolabel/weights/best.pt"
    MODEL_PATH3="/ultralytics001/yolo_obb/weight/3biaopan_label/weights/best.pt"
    MODEL_PATH4="/ultralytics001/yolo_obb/weight/4read/weights/best.pt"
    TXT_LOG_PATH="/ultralytics001/yolo_obb/Result_pointer.txt"
    UPLOAD_FOLDER="/ultralytics001/uploads"
    OUTPUT_FOLDER="/ultralytics001/outputs"
    _SECRET_KEY="bishe_flask_secret_2025"

    # 添加默认值(二维码错误处理分支)
    WECHAT_MINIPROGRAM_PATH = "pages/index/index"
    WECHAT_MINIPROGRAM_APPID = "your_miniprogram_appid"
    APK_DOWNLOAD_URL = "https://your-domain.com/downloads/app.apk"

app = Flask(__name__)
app.secret_key = _SECRET_KEY

JWT_SECRET = _SECRET_KEY
JWT_EXPIRATION = timedelta(days=180)  # Token过期时间，按需调整

CORS(app) 

# 数据库配置
DB_CONFIG = {
    "host":        _DB_HOST,
    "port":        _DB_PORT,
    "user":        _DB_USER,
    "password":    _DB_PASSWORD,
    "database":    _DB_NAME,
    "charset":     "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit":  False,
    "connect_timeout": 5,
}

def get_db():
    """获取数据库连接，密码错了或者服务没开会在这报错"""
    try:
        return pymysql.connect(**DB_CONFIG)
    except pymysql.err.OperationalError as e:
        code = e.args[0] if e.args else 0
        if code == 1045:
            hint = f"密码错了，去config.py改DB_PASSWORD，当前是'{_DB_PASSWORD}'"
        elif code in (2003, 2002):
            hint = f"MySQL没开或者端口{_DB_PORT}连不上，先运行check_db.py看看"
        else:
            hint = str(e)
        raise RuntimeError(f"数据库连接失败：{hint}") from e

def create_token(user_id, username, user_level):
    """生成JWT Token"""
    import datetime
    payload = {
        'user_id': user_id,
        'username': username,
        'user_level': user_level,
        'exp': datetime.datetime.utcnow() + JWT_EXPIRATION,
        'iat': datetime.datetime.utcnow()
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    return token

# HS256 是一种 JWT 签名算法，全称是 HMAC-SHA256。
# 作用
# 用来加密签名 JWT token，确保 token：

# 没有被篡改（完整性）
# 确实是你的服务器签发的（真实性）
def verify_token(token):
    """验证JWT Token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    
def require_token(f):
    """验证 JWT Token 的装饰器（小程序接口用）"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({"error": "缺少 Authorization header"}), 401
        
        try:
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
            else:
                token = auth_header
        except:
            return jsonify({"error": "Token 格式错误"}), 401
        
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            request.current_user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token 已过期，请重新登录"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Token 无效"}), 401
        except Exception as e:
            return jsonify({"error": f"Token 验证失败: {str(e)}"}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function

# 全局异常处理，api接口统一返回json
@app.errorhandler(Exception)
def handle_any_exception(e):
    tb = traceback.format_exc()
    print(f"[未捕获异常] {tb}")
    if request.path.startswith("/api/") or request.method == "POST":
        msg = str(e)
        if "MySQL" in msg or "Can't connect" in msg or "Connection refused" in msg:
            msg = "数据库连接失败，检查MySQL是否启动"
        return jsonify({"error": msg}), 500
    return f"<pre>{tb}</pre>", 500

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "接口不存在"}), 404
    return redirect(url_for("login_page"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# werkzeug自带哈希，不用另装bcrypt
def _hash_pw(plain: str) -> str:
    return generate_password_hash(plain)

def _check_pw(plain: str, hashed: str) -> bool:
    return check_password_hash(hashed, plain)


# =====  数据库初始化  =====
def init_db():
    print("初始化数据库...")
    # 先不指定database，确保BISHE库存在
    cfg_no_db = {k: v for k, v in DB_CONFIG.items()
                 if k not in ("database", "cursorclass", "autocommit")}
    cfg_no_db["connect_timeout"] = 5
    try:
        conn = pymysql.connect(**cfg_no_db, cursorclass=pymysql.cursors.DictCursor)
        with conn.cursor() as cur:
            cur.execute(
                "CREATE DATABASE IF NOT EXISTS `BISHE` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"建库失败：{e}")
        print("手动执行：CREATE DATABASE BISHE DEFAULT CHARSET utf8mb4;")
        return

    try:
        conn = get_db()
    except RuntimeError as e:
        print(f"连接失败：{e}")
        return

    try:
        with conn.cursor() as cur:
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
            # # 默认管理员
            # cur.execute("SELECT id FROM `user` WHERE username='admin' LIMIT 1")
            # if not cur.fetchone():
            #     cur.execute(
            #         "INSERT INTO `user`(username,password,user_level) VALUES(%s,%s,%s)",
            #         ("admin", _hash_pw("admin123"), "super_admin")
            #     )
            #     print("已创建默认账号 admin / admin123")
        conn.commit()
        print("数据库初始化完成")
    except Exception as e:
        print(f"建表失败：{e}")
    finally:
        conn.close()


# 读序号对照表
try:
    _df = pd.read_excel(FILE_EXCEL_PATH, engine="openpyxl")
    _df["序号"] = _df["序号"].astype(str)
    MY_DICT: dict = pd.Series(_df["表计"].values, index=_df["序号"]).to_dict()
    print(f"序号对照表加载成功，共{len(MY_DICT)}条")
except Exception as e:
    print(f"序号对照表加载失败：{e}")
    MY_DICT = {}


# task_id 生成：格式 YYYY-MM-DD-N
_date_counter: dict = {}
# 创建线程锁（防止并发冲突）
task_id_lock = threading.Lock()

# def generate_unique_task_id(conn):
#     """生成唯一的 task_id（线程安全）"""
#     with task_id_lock:  # 加锁
#         today = datetime.datetime.now().strftime('%Y-%m-%d')
        
#         with conn.cursor() as cur:
#             # 查询今天已有的最大序号
#             cur.execute(
#                 "SELECT task_id FROM yolo "
#                 "WHERE task_id LIKE %s "
#                 "ORDER BY LENGTH(task_id) DESC, task_id DESC LIMIT 1",
#                 (f"{today}-%",)
#             )
#             result = cur.fetchone()
            
#             if result:
#                 # 提取序号部分
#                 last_task_id = result['task_id']
#                 try:
#                     last_seq = int(last_task_id.split('-')[-1])
#                     next_seq = last_seq + 1
#                 except:
#                     next_seq = 1
#             else:
#                 next_seq = 1
            
#             # 生成新的 task_id
#             task_id = f"{today}-{next_seq}"
            
#             return task_id
def generate_unique_task_id(conn, user_id):
    """生成唯一的 task_id（线程安全，按用户递增，并修复了超过10的排序bug）"""
    with task_id_lock:  # 加锁
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        # 加上用户标识，例如 2026-05-20-U5-
        prefix = f"{today}-U{user_id}-"
        
        with conn.cursor() as cur:
            # 查询该用户今天已有的最大序号
            # 关键修改：加上 LENGTH(task_id) DESC，修复9大于10的bug
            cur.execute(
                "SELECT task_id FROM yolo "
                "WHERE task_id LIKE %s "
                "ORDER BY LENGTH(task_id) DESC, task_id DESC LIMIT 1",
                (f"{prefix}%",)
            )
            result = cur.fetchone()
            
            if result:
                # 提取序号部分 (以 '-' 分割取最后一段)
                last_task_id = result['task_id']
                try:
                    last_seq = int(last_task_id.split('-')[-1])
                    next_seq = last_seq + 1
                except:
                    next_seq = 1
            else:
                next_seq = 1
            
            # 生成新的专属 task_id，例如：2026-05-20-U5-1
            task_id = f"{prefix}{next_seq}"
            
            return task_id
        

# 内存里存任务状态（重启会丢，但检测一般很快，够用）
_task_states: dict = {}
_task_lock = threading.Lock()

def _fwdpath(p: str) -> str:
    # Windows路径用反斜杠，放到URL里会出问题，统一换成正斜杠
    # 写了三个replace感觉有点蠢但懒得改了，能用就行
    return p.replace('\\', '/').replace('\\', '/') if p else p


# 数学工具
def calc_intersection(p1, p2, p3, p4):
    # 求两条线段所在直线的交点
    try:
        m1=(p2[1]-p1[1])/(p2[0]-p1[0]); b1=p1[1]-m1*p1[0]
        m2=(p4[1]-p3[1])/(p4[0]-p3[0]); b2=p3[1]-m2*p3[0]
        if m1==m2: return None  # 平行
        x=(b2-b1)/(m1-m2); return (x, m1*x+b1)
    except ZeroDivisionError: return None  # 垂直线

def dist(a, b):
    return int(sqrt((a[0]-b[0])**2+(a[1]-b[1])**2))

def clock_angle(v1, v2):
    # 计算顺时针角度
    n=np.linalg.norm(v1)*np.linalg.norm(v2)
    rho=np.rad2deg(np.arcsin(np.clip(np.cross(v1,v2)/n,-1,1)))
    theta=np.rad2deg(np.arccos(np.clip(np.dot(v1,v2)/n,-1,1)))
    return theta if rho>0 else 360-theta

def mid_point(x1,y1,x2,y2,x3,y3,x4,y4):

    # 取OBB四条边里最短的两条的中点，用来估算OBB的长轴方向
    edges=[((x1,y1),(x2,y2)),((x2,y2),(x3,y3)),((x3,y3),(x4,y4)),((x4,y4),(x1,y1))]
    el=[(e,math.hypot(e[1][0]-e[0][0],e[1][1]-e[0][1])) for e in edges]
    el.sort(key=lambda x:x[1])
    mid=lambda p,q:((p[0]+q[0])/2,(p[1]+q[1])/2)
    return mid(*el[0][0]), mid(*el[1][0])

# def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10):
#     """绘制虚线
    
#     Args:
#         img: 图像
#         pt1: 起点 (x, y)
#         pt2: 终点 (x, y)
#         color: 颜色 (B, G, R)
#         thickness: 线宽
#         dash_length: 虚线段长度
#     """
#     dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
#     dashes = int(dist / dash_length)
    
#     for i in range(dashes):
#         if i % 2 == 0:  # 只画偶数段，实现虚线效果
#             start_ratio = i / dashes
#             end_ratio = (i + 1) / dashes
#             start = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
#                     int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
#             end = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
#                   int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))
#             cv2.line(img, start, end, color, thickness, cv2.LINE_AA)

def draw_dashed_line(img, pt1, pt2, color,
                     thickness=2,
                     dash_length=10,
                     extend_length=0):
    """
    绘制可延长虚线

    Args:
        img: 图像
        pt1: 起点
        pt2: 终点
        color: BGR颜色
        thickness: 线宽
        dash_length: 虚线长度
        extend_length: 超出 pt2 的延长长度
    """

    x1, y1 = pt1
    x2, y2 = pt2

    # 原始方向向量
    dx = x2 - x1
    dy = y2 - y1

    dist = math.hypot(dx, dy)

    if dist == 0:
        return

    # 单位方向向量
    ux = dx / dist
    uy = dy / dist

    # 延长终点
    x2_ext = x2 + ux * extend_length
    y2_ext = y2 + uy * extend_length

    total_dist = math.hypot(x2_ext - x1, y2_ext - y1)

    dashes = int(total_dist / dash_length)

    for i in range(dashes):
        if i % 2 == 0:

            start_ratio = i / dashes
            end_ratio = (i + 1) / dashes

            start = (
                int(x1 + (x2_ext - x1) * start_ratio),
                int(y1 + (y2_ext - y1) * start_ratio)
            )

            end = (
                int(x1 + (x2_ext - x1) * end_ratio),
                int(y1 + (y2_ext - y1) * end_ratio)
            )

            cv2.line(img, start, end, color, thickness, cv2.LINE_AA)


# DB辅助：更新yolo表
def db_update_yolo(task_id: str, **fields):
    if not fields: return
    set_clause = ", ".join(f"`{k}`=%s" for k in fields)
    vals = list(fields.values()) + [task_id]
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE `yolo` SET {set_clause} WHERE task_id=%s", vals)
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"yolo更新失败 task_id={task_id}：{e}")


# 核心检测流程（跑在后台线程里）
def _run_detection(task_id, image_path, image_name, image_name_all, user_id):
    def log(msg):
        with _task_lock:
            _task_states[task_id]["logs"].append(msg)

    state = _task_states[task_id]
    state["status"] = "running"
    # db_update_yolo(task_id, detect_status="running")
    db_update_yolo(task_id, user_id=user_id, detect_status="running")

    try:
        rdir = os.path.join(OUTPUT_FOLDER,
               f"outputs-{datetime.datetime.now().strftime('%Y-%m-%d')}")
        rimg = os.path.join(rdir, image_name)
        os.makedirs(rimg, exist_ok=True)
        txt1 = os.path.join(rimg, "result.txt")

        # 步骤1：检测整个仪表区域并裁剪
        log("步骤1/5：裁剪仪表盘区域...")
        r1   = YOLO(MODEL_PATH1)(source=image_path, save=True, save_txt=True, save_crop=True, conf=0.7)
        sp1  = str(r1[0].save_dir)
        i1   = cv2.imread(os.path.join(sp1,"crops","Instrument",image_name_all))
        obb_path = os.path.join(rimg, image_name+"_all.jpg")
        cv2.imwrite(obb_path, i1)
        state["img_obb"] = _fwdpath(obb_path)
        db_update_yolo(task_id, obb_img_path=_fwdpath(obb_path))
        log("步骤1完成")
        imgp = obb_path

        # 步骤2：裁剪表盘（没有刻度标签的那一块）
        log("步骤2/5：裁剪表盘...")
        r2   = YOLO(MODEL_PATH2)(source=imgp, save=True, save_txt=True, save_crop=True, conf=0.7)
        sp2  = str(r2[0].save_dir)
        i2   = cv2.imread(os.path.join(sp2,"crops","Pointer",image_name+"_all.jpg"))
        dial_path = os.path.join(rimg, image_name+"_biaopan.jpg")
        cv2.imwrite(dial_path, i2)
        state["img_dial"] = _fwdpath(dial_path)
        db_update_yolo(task_id, dial_img_path=_fwdpath(dial_path))
        log("步骤2完成")

        # 步骤3：裁剪序号标签区域
        log("步骤3/5：裁剪序号标签...")
        r3   = YOLO(MODEL_PATH3)(source=imgp, save=True, save_txt=True, save_crop=True, conf=0.7)
        sp3  = str(r3[0].save_dir)
        i3   = cv2.imread(os.path.join(sp3,"crops","Label",image_name+"_all.jpg"))
        lbl_path = os.path.join(rimg, image_name+"_biaoqian.jpg")
        cv2.imwrite(lbl_path, i3)
        db_update_yolo(task_id, label_img_path=lbl_path)
        log("步骤3完成")

        # 步骤4：识别序号
        log("步骤4/5：识别序号标签...")
        r4   = YOLO(MODEL_PATH4)(source=lbl_path, save=True, save_txt=True, conf=0.7)
        lp4  = os.path.join(str(r4[0].save_dir), "labels")
        log("步骤4完成")

        # 步骤5：OBB检测 + 读数计算（最复杂的部分）
        log("步骤5/5：OBB关键点检测与读数计算...")
        r5   = YOLO(MODEL_PATH4)(source=dial_path, save=True, save_txt=True, conf=0.7)
        sp5  = str(r5[0].save_dir)
        lp5  = os.path.join(sp5, "labels")

        image = cv2.imread(os.path.join(sp5, image_name+"_biaopan.jpg"))
        h, w  = image.shape[:2]

        # 读检测结果txt，每行是一个OBB的坐标
        rows = []
        with open(os.path.join(lp5, image_name+"_biaopan.txt")) as f:
            for line in f: rows.append(line.strip().split())
        sr = sorted(rows, key=lambda x:(float(x[0]),float(x[1])))

        # 解析坐标，归一化坐标*宽高还原成像素坐标
        def rc(r,i): return float(r[i])
        xs1,ys1=w*rc(sr[0],7),h*rc(sr[0],8); xs2,ys2=w*rc(sr[0],1),h*rc(sr[0],2)
        xs3,ys3=w*rc(sr[0],3),h*rc(sr[0],4); xs4,ys4=w*rc(sr[0],5),h*rc(sr[0],6)
        xsf,ysf=(xs1+xs2+xs3+xs4)/4,(ys1+ys2+ys3+ys4)/4  # 起点OBB中心

        xe1,ye1=w*rc(sr[1],1),h*rc(sr[1],2); xe2,ye2=w*rc(sr[1],3),h*rc(sr[1],4)
        xe3,ye3=w*rc(sr[1],5),h*rc(sr[1],6); xe4,ye4=w*rc(sr[1],7),h*rc(sr[1],8)
        xef,yef=(xe1+xe2+xe3+xe4)/4,(ye1+ye2+ye3+ye4)/4  # 终点OBB中心

        xp1,yp1=w*rc(sr[3],7),h*rc(sr[3],8); xp2,yp2=w*rc(sr[3],1),h*rc(sr[3],2)
        xp3,yp3=w*rc(sr[3],3),h*rc(sr[3],4); xp4,yp4=w*rc(sr[3],5),h*rc(sr[3],6)
        xpf,ypf=(xp1+xp2+xp3+xp4)/4,(yp1+yp2+yp3+yp4)/4  # 指针OBB中心

        xz1,yz1=w*rc(sr[2],1),h*rc(sr[2],2); xz2,yz2=w*rc(sr[2],3),h*rc(sr[2],4)
        xz3,yz3=w*rc(sr[2],5),h*rc(sr[2],6); xz4,yz4=w*rc(sr[2],7),h*rc(sr[2],8)
        xzf,yzf=(xz1+xz2+xz3+xz4)/4,(yz1+yz2+yz3+yz4)/4  # 零点OBB中心

        (ss1,ss2),(ss3,ss4)=mid_point(xs1,ys1,xs2,ys2,xs3,ys3,xs4,ys4)
        (ee1,ee2),(ee3,ee4)=mid_point(xe1,ye1,xe2,ye2,xe3,ye3,xe4,ye4)
        (pp1,pp2),(pp3,pp4)=mid_point(xp1,yp1,xp2,yp2,xp3,yp3,xp4,yp4)
        (zz1,zz2),(zz3,zz4)=mid_point(xz1,yz1,xz2,yz2,xz3,yz3,xz4,yz4)
        p1=(ss1,ss2);p2=(ss3,ss4);p3=(ee1,ee2);p4=(ee3,ee4)
        p5=(pp1,pp2);p6=(pp3,pp4);p7=(zz1,zz2);p8=(zz3,zz4)

        # 三对直线两两求交点，取最近两个的中点作为圆心估计
        # 这个方法是参考论文里的做法，三条线不会严格交于一点所以取中点
        ise=calc_intersection(p1,p2,p3,p4); isz=calc_intersection(p1,p2,p7,p8); iez=calc_intersection(p3,p4,p7,p8)
        d44=min(dist(ise,isz),dist(ise,iez),dist(isz,iez))
        if d44==dist(ise,isz):   cxx=(ise[0]+isz[0])/2;cyy=(ise[1]+isz[1])/2
        elif d44==dist(ise,iez): cxx=(ise[0]+iez[0])/2;cyy=(ise[1]+iez[1])/2
        else:                     cxx=(isz[0]+iez[0])/2;cyy=(isz[1]+iez[1])/2

        ise2=calc_intersection(p1,p2,p3,p4); isp=calc_intersection(p1,p2,p5,p6); iep=calc_intersection(p3,p4,p5,p6)
        d4=min(dist(ise2,isp),dist(ise2,iep),dist(isp,iep))
        if d4==dist(ise2,isp):   cx=(ise2[0]+isp[0])/2;cy=(ise2[1]+isp[1])/2
        elif d4==dist(ise2,iep): cx=(ise2[0]+iep[0])/2;cy=(ise2[1]+iep[1])/2
        else:                     cx=(isp[0]+iep[0])/2; cy=(isp[1]+iep[1])/2

        # 在图上画出关键点，方便调试
        for pt,col in [((int(ise2[0]),int(ise2[1])),(0,0,255)),((int(isp[0]),int(isp[1])),(0,0,120)),
                       ((int(iep[0]),int(iep[1])),(0,0,0)),((int(cx),int(cy)),(0,0,255)),
                       ((int(xzf),int(yzf)),(0,0,255)),((int(cxx),int(cyy)),(0,0,255))]:
            cv2.circle(image,pt,5,col,-1)

        # 保存拟合结果图
        fit_path = os.path.join(rimg, image_name + "_fitting.jpg")
        write_ok = cv2.imwrite(fit_path, image)
        if not write_ok:
            # 有时候相对路径在某些环境下写不进去，改用绝对路径试试
            fit_path_abs = os.path.abspath(fit_path)
            write_ok = cv2.imwrite(fit_path_abs, image)
            if write_ok:
                fit_path = fit_path_abs
                print(f"绝对路径写入成功：{fit_path}")
        else:
            print(f"拟合图写入失败：{fit_path}")

        if write_ok and os.path.isfile(fit_path) and os.path.getsize(fit_path) > 0:
            state["img_fitting"] = _fwdpath(fit_path)
            print(f"拟合图保存成功：{state['img_fitting']}")
        else:
            print(f"拟合图文件异常：{fit_path}")

        #  生成 fitLine 可视化图
        # 重新读取原始表盘图像（不带fitting标注的）
        image_fitline = cv2.imread(dial_path)
        
        # 定义颜色：红色用于中点，浅紫色用于轴线
        red_color = (0, 0, 255)  # BGR格式的红色
        light_purple = (220, 160, 220)  # BGR格式的浅紫色
        
        # 1. 标定所有短边几何中心（红点）
        midpoints = [
            (int(ss1), int(ss2)),  # 起点S的第一个中点
            (int(ss3), int(ss4)),  # 起点S的第二个中点
            (int(ee1), int(ee2)),  # 终点E的第一个中点
            (int(ee3), int(ee4)),  # 终点E的第二个中点
            (int(pp1), int(pp2)),  # 指针P的第一个中点
            (int(pp3), int(pp4)),  # 指针P的第二个中点
            (int(zz1), int(zz2)),  # 零点Z的第一个中点
            (int(zz3), int(zz4))   # 零点Z的第二个中点
        ]
        
        # 在图像上画红色圆点标记中点
        for pt in midpoints:
            cv2.circle(image_fitline, pt, 6, red_color, -1)  # 红色实心圆，半径6
        
        # 2. 生成贯穿指针、起点、终点、零点的中心线（浅紫色轴线）
        # 连接每组的两个中点形成轴线
        lines = [
            ((int(ss1), int(ss2)), (int(ss3), int(ss4))),  # 起点S的轴线
            ((int(ee1), int(ee2)), (int(ee3), int(ee4))),  # 终点E的轴线
            ((int(pp1), int(pp2)), (int(pp3), int(pp4))),  # 指针P的轴线
            ((int(zz1), int(zz2)), (int(zz3), int(zz4)))   # 零点Z的轴线
        ]
        
        # 画浅紫色轴线
        for line in lines:
            cv2.line(image_fitline, line[0], line[1], light_purple, 3)  # 线宽3
        
        # 保存 fitLine 图像
        fitline_path = os.path.join(rimg, image_name + "_fitLine.jpg")
        write_fitline_ok = cv2.imwrite(fitline_path, image_fitline)
        
        if not write_fitline_ok:
            # 尝试使用绝对路径
            fitline_path_abs = os.path.abspath(fitline_path)
            write_fitline_ok = cv2.imwrite(fitline_path_abs, image_fitline)
            if write_fitline_ok:
                fitline_path = fitline_path_abs
                print(f"fitLine图绝对路径写入成功：{fitline_path}")
        
        if write_fitline_ok and os.path.isfile(fitline_path) and os.path.getsize(fitline_path) > 0:
            state["img_fitline"] = _fwdpath(fitline_path)
            print(f"fitLine图保存成功：{state['img_fitline']}")
        else:
            print(f"fitLine图写入失败：{fitline_path}")


        # 计算读数
        # rv1：方法一（用指针圆心），rv2：方法二（用平均圆心），final：加权修正后的最终结果
        rv1=clock_angle([xsf-cx,ysf-cy],[xpf-cx,ypf-cy])/clock_angle([xsf-cx,ysf-cy],[xef-cx,yef-cy])-0.1
        acx=(cx+cxx)/2; acy=(cy+cyy)/2
        th3=clock_angle([xsf-acx,ysf-acy],[xpf-acx,ypf-acy])
        th4=clock_angle([xsf-acx,ysf-acy],[xef-acx,yef-acy])
        rv2=th3/th4-0.1
        rv3=clock_angle([xsf-acx,ysf-acy],[xzf-acx,yzf-acy])/th4-0.1
        final=rv2+(0.4-rv3)/2  # 修正公式，0.4是经验值，跑了几十张图调出来的


        # 生成 fitCenter 可视化图
        # 重新读取原始表盘图像
        image_fitcenter = cv2.imread(dial_path)
        
        # 定义颜色
        orange_yellow = (0, 165, 255)  # BGR格式的橙黄色
        pink = (203, 192, 255)  # BGR格式的粉红色
        red_dash = (0, 0, 255)  # BGR格式的红色（用于虚线）
        
        # 计算圆的半径（使用圆心到起点的距离）
        radius_cx = int(math.sqrt((xsf - cx)**2 + (ysf - cy)**2))
        radius_acx = int(math.sqrt((xsf - acx)**2 + (ysf - acy)**2))
        
        # 1. 绘制指针圆心(cx, cy)相关的内容
        # 绘制辅助轴线（红色虚线）
        # S轴线：p1=(ss1,ss2), p2=(ss3,ss4)
        draw_dashed_line(image_fitcenter, (int(ss1), int(ss2)), (int(ss3), int(ss4)), red_dash, 2, 10)
        # E轴线：p3=(ee1,ee2), p4=(ee3,ee4)
        draw_dashed_line(image_fitcenter, (int(ee1), int(ee2)), (int(ee3), int(ee4)), red_dash, 2, 10)
        # P轴线：p5=(pp1,pp2), p6=(pp3,pp4)
        draw_dashed_line(image_fitcenter, (int(pp1), int(pp2)), (int(pp3), int(pp4)), red_dash, 2, 10)
        
        # 绘制S-E, S-P, E-P三个交点（小红点）
        cv2.circle(image_fitcenter, (int(ise2[0]), int(ise2[1])), 4, red_dash, -1)
        cv2.circle(image_fitcenter, (int(isp[0]), int(isp[1])), 4, red_dash, -1)
        cv2.circle(image_fitcenter, (int(iep[0]), int(iep[1])), 4, red_dash, -1)
        
        # # 绘制从选中的两个交点到指针圆心的连线（红色虚线）
        # if d4 == dist(ise2, isp):
        #     draw_dashed_line(image_fitcenter, (int(ise2[0]), int(ise2[1])), (int(cx), int(cy)), red_dash, 1, 8)
        #     draw_dashed_line(image_fitcenter, (int(isp[0]), int(isp[1])), (int(cx), int(cy)), red_dash, 1, 8)
        # elif d4 == dist(ise2, iep):
        #     draw_dashed_line(image_fitcenter, (int(ise2[0]), int(ise2[1])), (int(cx), int(cy)), red_dash, 1, 8)
        #     draw_dashed_line(image_fitcenter, (int(iep[0]), int(iep[1])), (int(cx), int(cy)), red_dash, 1, 8)
        # else:
        #     draw_dashed_line(image_fitcenter, (int(isp[0]), int(isp[1])), (int(cx), int(cy)), red_dash, 1, 8)
        #     draw_dashed_line(image_fitcenter, (int(iep[0]), int(iep[1])), (int(cx), int(cy)), red_dash, 1, 8)

        # 绘制从选中的两个交点到指针圆心的连线（红色虚线）
        if d4 == dist(ise2, isp):

            draw_dashed_line(
                image_fitcenter,
                (int(ise2[0]), int(ise2[1])),
                (int(cx), int(cy)),
                red_dash,
                thickness=1,
                dash_length=8,
                extend_length=40
            )

            draw_dashed_line(
                image_fitcenter,
                (int(isp[0]), int(isp[1])),
                (int(cx), int(cy)),
                red_dash,
                thickness=1,
                dash_length=8,
                extend_length=40
            )

        elif d4 == dist(ise2, iep):

            draw_dashed_line(
                image_fitcenter,
                (int(ise2[0]), int(ise2[1])),
                (int(cx), int(cy)),
                red_dash,
                thickness=1,
                dash_length=8,
                extend_length=40
            )

            draw_dashed_line(
                image_fitcenter,
                (int(iep[0]), int(iep[1])),
                (int(cx), int(cy)),
                red_dash,
                thickness=1,
                dash_length=8,
                extend_length=40
            )

        else:

            draw_dashed_line(
                image_fitcenter,
                (int(isp[0]), int(isp[1])),
                (int(cx), int(cy)),
                red_dash,
                thickness=1,
                dash_length=8,
                extend_length=40
            )

            draw_dashed_line(
                image_fitcenter,
                (int(iep[0]), int(iep[1])),
                (int(cx), int(cy)),
                red_dash,
                thickness=1,
                dash_length=8,
                extend_length=40
            )
            
        
        # 绘制指针圆心的完整圆周（橙黄色）
        cv2.circle(image_fitcenter, (int(cx), int(cy)), radius_cx, orange_yellow, 3)
        
        # 标注指针圆心（橙黄色实心圆）
        cv2.circle(image_fitcenter, (int(cx), int(cy)), 8, orange_yellow, -1)
        
        # 2. 绘制零点圆心(cxx, cyy)相关的内容
        # Z轴线：p7=(zz1,zz2), p8=(zz3,zz4)（红色虚线）
        draw_dashed_line(image_fitcenter, (int(zz1), int(zz2)), (int(zz3), int(zz4)), red_dash, 2, 10)
        
        # 绘制S-E, S-Z, E-Z三个交点（小红点）
        cv2.circle(image_fitcenter, (int(ise[0]), int(ise[1])), 4, red_dash, -1)
        cv2.circle(image_fitcenter, (int(isz[0]), int(isz[1])), 4, red_dash, -1)
        cv2.circle(image_fitcenter, (int(iez[0]), int(iez[1])), 4, red_dash, -1)
        
        # 绘制从选中的两个交点到零点圆心的连线（红色虚线）
        if d44 == dist(ise, isz):
            draw_dashed_line(image_fitcenter, (int(ise[0]), int(ise[1])), (int(cxx), int(cyy)), red_dash, 1, 8)
            draw_dashed_line(image_fitcenter, (int(isz[0]), int(isz[1])), (int(cxx), int(cyy)), red_dash, 1, 8)
        elif d44 == dist(ise, iez):
            draw_dashed_line(image_fitcenter, (int(ise[0]), int(ise[1])), (int(cxx), int(cyy)), red_dash, 1, 8)
            draw_dashed_line(image_fitcenter, (int(iez[0]), int(iez[1])), (int(cxx), int(cyy)), red_dash, 1, 8)
        else:
            draw_dashed_line(image_fitcenter, (int(isz[0]), int(isz[1])), (int(cxx), int(cyy)), red_dash, 1, 8)
            draw_dashed_line(image_fitcenter, (int(iez[0]), int(iez[1])), (int(cxx), int(cyy)), red_dash, 1, 8)
        
        # 标注零点圆心（紫色实心圆，作为辅助参考）
        cv2.circle(image_fitcenter, (int(cxx), int(cyy)), 6, (255, 0, 255), -1)
        
        # 3. 绘制平均圆心(acx, acy)相关的内容
        # 绘制从两个圆心到平均圆心的连线（红色虚线）
        draw_dashed_line(image_fitcenter, (int(cx), int(cy)), (int(acx), int(acy)), red_dash, 2, 8)
        draw_dashed_line(image_fitcenter, (int(cxx), int(cyy)), (int(acx), int(acy)), red_dash, 2, 8)
        
        # 绘制平均圆心的完整圆周（粉红色）
        cv2.circle(image_fitcenter, (int(acx), int(acy)), radius_acx, pink, 3)
        
        # 标注平均圆心（粉红色实心圆）
        cv2.circle(image_fitcenter, (int(acx), int(acy)), 8, pink, -1)
        
        # 添加文字标注（可选）
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_fitcenter, 'Pointer Center', (int(cx) + 15, int(cy) - 10), 
                   font, 0.5, orange_yellow, 2, cv2.LINE_AA)
        cv2.putText(image_fitcenter, 'Average Center', (int(acx) + 15, int(acy) + 20), 
                   font, 0.5, pink, 2, cv2.LINE_AA)
        
        # 保存 fitCenter 图像
        fitcenter_path = os.path.join(rimg, image_name + "_fitCenter.jpg")
        write_fitcenter_ok = cv2.imwrite(fitcenter_path, image_fitcenter)
        
        if not write_fitcenter_ok:
            # 尝试使用绝对路径
            fitcenter_path_abs = os.path.abspath(fitcenter_path)
            write_fitcenter_ok = cv2.imwrite(fitcenter_path_abs, image_fitcenter)
            if write_fitcenter_ok:
                fitcenter_path = fitcenter_path_abs
                print(f"fitCenter图绝对路径写入成功：{fitcenter_path}")
        
        if write_fitcenter_ok and os.path.isfile(fitcenter_path) and os.path.getsize(fitcenter_path) > 0:
            state["img_fitcenter"] = _fwdpath(fitcenter_path)
            print(f"fitCenter图保存成功：{state['img_fitcenter']}")
        else:
            print(f"fitCenter图写入失败：{fitcenter_path}")


        # 查序号对照表
        snum="未知"
        bq_txt=os.path.join(lp4, image_name+"_biaoqian.txt")
        with open(bq_txt) as f:
            for line in f:
                snum=MY_DICT.get(line.split()[0], f"序号{line.split()[0]}")
                break  # 只取第一行

        now=datetime.datetime.now()
        op=state.get("operator","unknown")
        entry=(f"{now.strftime('%Y-%m-%d %H:%M:%S')}\n任务编号：{task_id}  操作人：{op}\n"
               f"{image_name} {snum}\n修正前读数为{rv1:.6f}\n修正后读数为{final:.6f}\n\n")
        for p in [TXT_LOG_PATH, txt1]:
            with open(p,"a",encoding="utf-8") as f: f.write(entry)

        db_update_yolo(task_id, serial_number=snum, reading_before=round(rv1,6),
                       reading_after=round(final,6), detect_status="success",
                       detected_at=now.strftime("%Y-%m-%d %H:%M:%S"))

        with _task_lock:
            state.update({"status":"done","detect_time":now.strftime("%Y-%m-%d %H:%M:%S"),
                          "serial_number":snum,"reading_before":round(rv1,6),"reading_after":round(final,6)})
        log(f"完成 | 序号：{snum}")
        log(f"修正前读数：{rv1:.6f}")
        log(f"修正后读数：{final:.6f}")

    except Exception as exc:
        traceback.print_exc()
        with _task_lock:
            _task_states[task_id]["status"]="error"
            _task_states[task_id]["error"]=str(exc)
        db_update_yolo(task_id, detect_status="failed")
        log(f"检测出错：{exc}")

def api_auth_required(f):
    """API认证装饰器 - 支持JWT Token和Session"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 尝试Bearer Token认证（微信小程序）
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            payload = verify_token(token)
            if payload:
                request.current_user_id = payload['user_id']
                request.current_username = payload['username']
                request.current_user_level = payload['user_level']
                return f(*args, **kwargs)
        
        # 尝试Session认证（Web端）
        if session.get('user_id'):
            request.current_user_id = session.get('user_id')
            request.current_username = session.get('username')
            request.current_user_level = session.get('user_level')
            return f(*args, **kwargs)
        
        return jsonify({"error": "未登录或登录已过期"}), 401
    
    return decorated_function

# Web端登录装饰器
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            if request.method == "GET":
                return redirect(url_for("login_page"))
            return jsonify({"error":"请先登录","redirect":"/login"}), 401
        return f(*args, **kwargs)
    return decorated


def get_json() -> dict:
    """安全获取请求body的json，解析失败返回空dict"""
    return request.get_json(force=True, silent=True) or {}

# # API认证装饰器（支持JWT Token和Session两种方式）
# def login_required_token(f):
#     """Token验证装饰器"""
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         token = request.headers.get('Authorization', '').replace('Bearer ', '')
#         if not token:
#             return jsonify({"error": "未授权"}), 401
        
#         payload = verify_token(token)
#         if not payload:
#             return jsonify({"error": "Token无效"}), 401
        
#         # 将用户信息注入到request对象
#         request.user_id = payload['user_id']
#         request.username = payload['username']
#         request.user_level = payload['user_level']
        
#         return f(*args, **kwargs)
#     return decorated_function

# 路由：认证
@app.route("/login")
def login_page():
    if "username" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    """微信小程序登录接口（返回JWT Token）"""
    data     = get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    try:
        conn = get_db()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password, user_level "
                "FROM `user` WHERE username=%s LIMIT 1",
                (username,)
            )
            user = cur.fetchone()
    except Exception as e:
        return jsonify({"error": f"查询失败：{e}"}), 500
    finally:
        conn.close()

    if not user or not _check_pw(password, user["password"]):
        return jsonify({"error": "用户名或密码错误"}), 401

    # 关键：先清空旧 session
    session.clear()

    # 再写入新用户
    session["user_id"]    = user["id"]
    session["username"]   = user["username"]
    session["user_level"] = user["user_level"]

    # （可选）调试用
    print("登录成功 user_id =", user["id"])

    return jsonify({
        "message":    "登录成功",
        "user_id":   user["id"],
        "username":   user["username"],
        "user_level": user["user_level"],
        "redirect":   "/"
    })


@app.route("/api/register", methods=["POST"])
def api_register():
    data       = get_json()
    username   = data.get("username", "").strip()
    password   = data.get("password", "").strip()
    user_level = data.get("user_level", "user")

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if len(username) < 3:
        return jsonify({"error": "用户名至少3个字符"}), 400
    if len(password) < 6:
        return jsonify({"error": "密码至少6个字符"}), 400
    if user_level not in ("super_admin", "admin", "user"):
        user_level = "user"  # 非法值直接当普通用户

    try:
        conn = get_db()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM `user` WHERE username=%s LIMIT 1", (username,))
            if cur.fetchone():
                return jsonify({"error": "用户名已存在"}), 409
            cur.execute(
                "INSERT INTO `user`(username, password, user_level) VALUES(%s, %s, %s)",
                (username, _hash_pw(password), user_level)
            )
        conn.commit()
    except Exception as e:
        return jsonify({"error": f"注册失败：{e}"}), 500
    finally:
        conn.close()

    # # 自动登录，返回Token
    # token = generate_token(user_id, username, "user")

    # return jsonify({
    #         "message": "注册成功",
    #         "token": token,
    #         "user_id": user_id,
    #         "username": username,
    #         "user_level": "user"
    #     })    

    return jsonify({"message": "注册成功，请登录"})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# 路由：主页面
@app.route("/")
@login_required
def index():
    return render_template("index.html",
                           username   = session.get("username"),
                           user_level = session.get("user_level"))

# # Web 端接口
# @app.route("/api/upload_task", methods=["POST"])
# def web_upload_task():
#     """Web 端创建任务"""
#     try:
#         conn = get_db()
        
#         # 生成唯一 task_id
#         task_id = generate_unique_task_id(conn)
        
#         # 插入数据
#         with conn.cursor() as cur:
#             cur.execute(
#                 "INSERT INTO yolo (task_id, user_id, image_path, created_at) "
#                 "VALUES (%s, %s, %s, NOW())",
#                 (task_id, session.get('user_id'), image_path)
#             )
#             conn.commit()
        
#         return jsonify({
#             "message": "提交成功",
#             "task_id": task_id
#         })
    
#     except pymysql.err.IntegrityError as e:
#         # 如果还是冲突，返回友好错误
#         return jsonify({"error": "任务ID冲突，请重试"}), 409
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         conn.close()
        
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "未收到文件"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    # 先获取数据库
    conn = get_db()
    user_id = session.get("user_id")
    task_id   = generate_unique_task_id(conn, user_id)  # 生成唯一的 task_id 

    ext       = os.path.splitext(f.filename)[1].lower()
    save_name = f"{task_id}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, save_name)
    f.save(save_path)

    user_id = session.get("user_id")
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO `yolo`(task_id,user_id,original_img_path,detect_status) "
                    "VALUES(%s,%s,%s,'pending')",
                    (task_id, user_id, save_path)
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"插入yolo记录失败：{e}")  # 这里失败了检测还能继续，先只打印

    with _task_lock:
        _task_states[task_id] = {
            "status":"uploaded","logs":[],
            "img_obb":None,"img_dial":None,"img_fitting":None,
            "detect_time":None,"serial_number":None,
            "reading_before":None,"reading_after":None,
            "image_path":save_path,"image_name":task_id,
            "image_name_all":save_name,"orig_filename":f.filename,
            "operator":session.get("username","unknown"),
        }

    return jsonify({"task_id":task_id,"image_url":f"/image/{save_path}","filename":f.filename})


@app.route("/upload_base64", methods=["POST"])
@login_required
def upload_base64():
    """接收摄像头拍的base64图片，和/upload逻辑基本一样"""
    import base64 as _b64
    data   = get_json()
    b64str = data.get("image", "").strip()
    if not b64str:
        return jsonify({"error": "未收到图片数据"}), 400

    # data URL前缀（"data:image/jpeg;base64,"）去掉
    if "," in b64str:
        b64str = b64str.split(",", 1)[1]

    try:
        img_bytes = _b64.b64decode(b64str)
    except Exception:
        return jsonify({"error": "base64解码失败"}), 400

    # task_id   = generate_task_id()
    # 先获取数据库
    # conn = get_db()
    # task_id   = generate_unique_task_id(conn)  # 生成唯一的 task_id 
    conn = get_db()
    # 注意：Web端路由里 user_id = session.get("user_id")
    # 小程序端路由里 user_id = request.current_user['user_id']
    user_id = session.get("user_id")
    task_id   = generate_unique_task_id(conn, user_id)

    save_name = f"{task_id}.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, save_name)

    try:
        with open(save_path, "wb") as fp:
            fp.write(img_bytes)
    except Exception as e:
        return jsonify({"error": f"图片保存失败：{e}"}), 500

    user_id = session.get("user_id")
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO `yolo`(task_id,user_id,original_img_path,detect_status) "
                    "VALUES(%s,%s,%s,'pending')",
                    (task_id, user_id, save_path)
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"插入yolo（base64）失败：{e}")

    with _task_lock:
        _task_states[task_id] = {
            "status":"uploaded","logs":[],
            "img_obb":None,"img_dial":None,"img_fitting":None,
            "detect_time":None,"serial_number":None,
            "reading_before":None,"reading_after":None,
            "image_path":save_path,"image_name":task_id,
            "image_name_all":save_name,"orig_filename":"camera_capture.jpg",
            "operator":session.get("username","unknown"),
        }

    return jsonify({"task_id":task_id,"image_url":f"/image/{save_path}","filename":"camera_capture.jpg"})


@app.route("/detect", methods=["POST"])
@login_required
def detect():
    data    = get_json()
    task_id = data.get("task_id")
    if not task_id or task_id not in _task_states:
        return jsonify({"error": "task_id无效"}), 400

    # 获取当前登录用户ID
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "用户未登录"}), 401

    st = _task_states[task_id]
    if st["status"] == "running":
        return jsonify({"error": "检测已在进行中"}), 400

    with _task_lock:
        st["status"] = "pending"; st["logs"] = ["开始检测..."]

    # 把 user_id 传进去
    threading.Thread(
        target=_run_detection,
        args=(task_id,
              st["image_path"],
              st["image_name"],
              st["image_name_all"],
              user_id,),
        daemon=True
    ).start()

    return jsonify({"task_id": task_id, "message": "检测已启动"})


@app.route("/poll/<task_id>")
@login_required
def poll(task_id):
    if task_id not in _task_states:
        return jsonify({"error": "task_id不存在"}), 404
    with _task_lock:
        st = dict(_task_states[task_id])
    img = lambda p: f"/image/{_fwdpath(p)}" if p else None
    return jsonify({
        "status":st["status"],"logs":st.get("logs",[]),
        "img_obb":img(st.get("img_obb")),"img_dial":img(st.get("img_dial")),
        "img_fitting":img(st.get("img_fitting")),
        "detect_time":st.get("detect_time"),"serial_number":st.get("serial_number"),
        "reading_before":st.get("reading_before"),"reading_after":st.get("reading_after"),
        "error":st.get("error"),
    })


@app.route("/confirm", methods=["POST"])
@login_required
def confirm():
    data    = get_json()
    task_id = data.get("task_id")
    now     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if task_id:
        db_update_yolo(task_id, is_confirmed=1, confirmed_at=now)
    with open(TXT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{session.get('username')}][{task_id}] confirmed. {now}\n")
    return jsonify({"message": "已确认"})


@app.route("/modify", methods=["POST"])
@login_required
def modify():
    data    = get_json()
    value   = data.get("value", "").strip()
    task_id = data.get("task_id")
    if not value:
        return jsonify({"error": "修改值不能为空"}), 400
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if task_id:
        try:
            db_update_yolo(task_id, reading_after=float(value))
        except (ValueError, TypeError):
            pass  # 输入不是数字就不更新，前端那边已经验证过了
    with open(TXT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{session.get('username')}][{task_id}] corrected:{value} {now}\n")
    return jsonify({"message": "已修改", "value": value})


@app.route("/clear", methods=["POST"])
@login_required
def clear():
    with open(TXT_LOG_PATH, "w", encoding="utf-8") as f: f.write("")
    return jsonify({"message": "日志已清除"})


@app.route("/get_log")
@login_required
def get_log():
    try:
        with open(TXT_LOG_PATH, "r", encoding="utf-8") as f: content = f.read()
    except FileNotFoundError:
        content = ""
    return jsonify({"content": content})


@app.route("/api/history")
@login_required
def api_history():
    user_id = session.get("user_id")
    page    = max(1, int(request.args.get("page", 1)))
    size    = min(50, int(request.args.get("size", 20)))
    offset  = (page - 1) * size
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT task_id,serial_number,reading_before,reading_after,"
                    "detect_status,is_confirmed,detected_at,created_at "
                    "FROM `yolo` WHERE user_id=%s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (user_id, size, offset)
                )
                rows = cur.fetchall()
                cur.execute("SELECT COUNT(*) AS total FROM `yolo` WHERE user_id=%s", (user_id,))
                total = cur.fetchone()["total"]
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # datetime不能直接json序列化，转成字符串
    records = []
    for r in rows:
        records.append({
            "task_id":       r["task_id"],
            "serial_number": r["serial_number"],
            "reading_before": float(r["reading_before"]) if r["reading_before"] is not None else None,
            "reading_after":  float(r["reading_after"])  if r["reading_after"]  is not None else None,
            "detect_status": r["detect_status"],
            "is_confirmed":  bool(r["is_confirmed"]),
            "detected_at":   str(r["detected_at"]) if r["detected_at"] else None,
            "created_at":    str(r["created_at"])  if r["created_at"]  else None,
        })
    return jsonify({"total": total, "page": page, "size": size, "records": records})


@app.route("/api/serial_history")
@login_required
def serial_history():
    serial = request.args.get("serial", "").strip()
    limit  = min(int(request.args.get("limit", 60)), 200)
    if not serial:
        return jsonify({"error": "serial参数不能为空"}), 400
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT task_id,reading_before,reading_after,detected_at,is_confirmed "
                    "FROM `yolo` WHERE serial_number=%s AND detect_status='success' "
                    "AND detected_at IS NOT NULL ORDER BY detected_at ASC LIMIT %s",
                    (serial, limit)
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    records = []
    for r in rows:
        records.append({
            "task_id":       r["task_id"],
            "reading_before": float(r["reading_before"]) if r["reading_before"] is not None else None,
            "reading_after":  float(r["reading_after"])  if r["reading_after"]  is not None else None,
            "detected_at":   str(r["detected_at"]) if r["detected_at"] else None,
            "is_confirmed":  bool(r["is_confirmed"]),
        })
    return jsonify({"serial": serial, "count": len(records), "records": records})

# ========== APK 下载路由 ==========
@app.route("/downloads/Android.apk")
def download_apk():
    """
    APK 文件下载路由
    提供本地 APK 文件下载
    """
    apk_dir = os.path.join(app.root_path, 'static', 'downloads')
    apk_filename = '仪态万象.apk'
    apk_path = os.path.join(apk_dir, apk_filename)
    
    # 检查文件是否存在
    if not os.path.exists(apk_path):
        return jsonify({
            "error": "APK 文件不存在",
            "hint": f"请将 APK 文件放置到: {apk_path}"
        }), 404
    
    # 返回文件下载
    return send_from_directory(
        apk_dir,
        apk_filename,
        as_attachment=True,  # 强制下载而不是在浏览器中打开
        download_name='仪态万象.apk',  # 下载时显示的文件名
        mimetype='application/vnd.android.package-archive'  # APK 的 MIME 类型
    )

@app.route("/api/qrcode")
def api_qrcode():
    """
    生成动态二维码
    微信扫描：跳转到微信小程序
    浏览器扫描：下载APK
    """
    return serve_qrcode(
        WECHAT_MINIPROGRAM_PATH, 
        WECHAT_MINIPROGRAM_APPID, 
        use_local_apk=True  # 使用本地 APK 下载
        )

@app.route("/api/qrcode/info")
def api_qrcode_info():
    """
    获取二维码信息（用于前端显示提示）
    """
    from flask import jsonify
    return jsonify(get_qrcode_info())

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('www', filename)

@app.route("/image/<path:filepath>")
# @login_required  # 图片访问不强制登录，保持和小程序端一致，改为API认证（支持Token和Session两种方式）
@api_auth_required
def serve_image(filepath):
    # Windows路径兼容，把反斜杠换掉
    # TODO: 感觉这块逻辑可以简化，之后有空再整理
    filepath = filepath.replace('\\', '/').replace('\\', '/').replace('\\', '/')
    abs_path = os.path.abspath(filepath)
    alt_path = os.path.abspath(filepath.replace('/', os.sep))
    allowed  = [os.path.abspath(OUTPUT_FOLDER), os.path.abspath(UPLOAD_FOLDER)]

    valid_path = None
    for candidate in [abs_path, alt_path]:
        if (any(candidate.startswith(a) for a in allowed)
                and os.path.isfile(candidate)):
            valid_path = candidate
            break

    if valid_path is None:
        print(f"路径访问失败：{filepath!r}")
        print(f"  abs={abs_path!r}")
        print(f"  allowed={allowed}")
        abort(404)

    return send_file(valid_path)

# 微信小程序专用API接口
@app.route("/api/wechat/login", methods=["POST"])
def wechat_login():
    """微信小程序登录接口（返回 JWT Token）"""
    data     = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    try:
        conn = get_db()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password, user_level "
                "FROM `user` WHERE username=%s LIMIT 1",
                (username,)
            )
            user = cur.fetchone()
    except Exception as e:
        return jsonify({"error": f"查询失败：{e}"}), 500
    finally:
        conn.close()

    if not user or not _check_pw(password, user["password"]):
        return jsonify({"error": "用户名或密码错误"}), 401

    # 小程序端：生成 JWT Token
    token = create_token(user["id"], user["username"], user["user_level"])

    return jsonify({
        "message":    "登录成功",
        "token":      token,  # 返回 token 给小程序
        "user_id":    user["id"],
        "username":   user["username"],
        "user_level": user["user_level"]
    })

@app.route("/api/wechat/register", methods=["POST"])
def wechat_register():
    """微信小程序注册接口（注册成功后返回 JWT Token）"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "未收到数据"}), 400
    
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    user_level = data.get("user_level", "user")

    # 验证用户名和密码
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if len(username) < 3:
        return jsonify({"error": "用户名至少3个字符"}), 400
    if len(password) < 6:
        return jsonify({"error": "密码至少6个字符"}), 400
    
    # 验证用户级别
    if user_level not in ("super_admin", "admin", "user"):
        user_level = "user"  # 非法值直接当普通用户

    conn = None
    try:
        conn = get_db()
        
        with conn.cursor() as cur:
            # 检查用户名是否已存在
            cur.execute(
                "SELECT id FROM `user` WHERE username=%s LIMIT 1",
                (username,)
            )
            if cur.fetchone():
                return jsonify({"error": "用户名已存在"}), 409
            
            # 插入新用户
            cur.execute(
                "INSERT INTO `user`(username, password, user_level) VALUES(%s, %s, %s)",
                (username, _hash_pw(password), user_level)
            )
            conn.commit()
            
            # 获取新插入的用户 ID
            new_user_id = cur.lastrowid
        
        # 注册成功后，自动生成 JWT Token（无需再次登录）
        token = create_token(new_user_id, username, user_level)
        
        return jsonify({
            "message": "注册成功",
            "token": token,
            "user_id": new_user_id,
            "username": username,
            "user_level": user_level
        }), 200 # 200 Created
    
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": f"注册失败：{str(e)}"}), 500
    
    finally:
        if conn:
            conn.close()

@app.route("/api/wechat/logout", methods=["POST"])
@require_token
def api_logout():
    session.clear()
    """微信小程序退出登录接口"""
    # JWT 是无状态的，服务端不需要做任何操作
    # 客户端删除 token 即可
    return jsonify({
        "message": "退出成功"
    }), 200

#  微信小程序：本地图片上传 
@app.route("/api/wechat/upload", methods=["POST"])
@require_token  # 使用 JWT 验证
def wechat_upload():
    """微信小程序：本地图片上传"""
    if "file" not in request.files:
        return jsonify({"error": "未收到文件"}), 400
    
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    conn = None
    try:
        conn = get_db()
        # 注意：Web端路由里 user_id = session.get("user_id")
        # 小程序端路由里 user_id = request.current_user['user_id']
        user_id = request.current_user['user_id']
        task_id   = generate_unique_task_id(conn, user_id)

        # 保存文件
        ext = os.path.splitext(f.filename)[1].lower()
        save_name = f"{task_id}{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, save_name)
        f.save(save_path)

        # 从 token 中获取用户信息（不是 session）
        user_id = request.current_user['user_id']
        username = request.current_user.get('username', 'unknown')

        # 插入数据库
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO `yolo`(task_id, user_id, original_img_path, detect_status) "
                    "VALUES(%s, %s, %s, 'pending')",
                    (task_id, user_id, save_path)
                )
            conn.commit()
        except Exception as e:
            print(f"插入yolo记录失败：{e}")
            return jsonify({"error": f"数据库插入失败：{e}"}), 500

        # 更新任务状态（与 Web 端相同）
        with _task_lock:
            _task_states[task_id] = {
                "status": "uploaded",
                "logs": [],
                "img_obb": None,
                "img_dial": None,
                "img_fitting": None,
                "detect_time": None,
                "serial_number": None,
                "reading_before": None,
                "reading_after": None,
                "image_path": save_path,
                "image_name": task_id,
                "image_name_all": save_name,
                "orig_filename": f.filename,
                "operator": username,
            }

        return jsonify({
            "message": "上传成功",
            "task_id": task_id,
            "image_url": f"/image/{save_path}",
            "filename": f.filename
        })

    except Exception as e:
        return jsonify({"error": f"上传失败：{str(e)}"}), 500
    
    finally:
        if conn:
            conn.close()


#  微信小程序：实时拍摄（base64）
@app.route("/api/wechat/upload_base64", methods=["POST"])
@require_token  # 使用 JWT 验证
def wechat_upload_base64():
    """微信小程序：接收摄像头拍的base64图片"""
    import base64 as _b64
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "未收到数据"}), 400
    
    b64str = data.get("image", "").strip()
    if not b64str:
        return jsonify({"error": "未收到图片数据"}), 400

    # 去掉 data URL 前缀（"data:image/jpeg;base64,"）
    if "," in b64str:
        b64str = b64str.split(",", 1)[1]

    # 解码 base64
    try:
        img_bytes = _b64.b64decode(b64str)
    except Exception:
        return jsonify({"error": "base64解码失败"}), 400

    conn = None
    try:
        # 获取数据库连接
        conn = get_db()
        user_id = request.current_user['user_id']
        task_id = generate_unique_task_id(conn, user_id)  # 生成唯一的 task_id

        # 保存图片
        save_name = f"{task_id}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, save_name)

        try:
            with open(save_path, "wb") as fp:
                fp.write(img_bytes)
        except Exception as e:
            return jsonify({"error": f"图片保存失败：{e}"}), 500

        # 从 token 中获取用户信息
        user_id = request.current_user['user_id']
        username = request.current_user.get('username', 'unknown')

        # 插入数据库
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO `yolo`(task_id, user_id, original_img_path, detect_status) "
                    "VALUES(%s, %s, %s, 'pending')",
                    (task_id, user_id, save_path)
                )
            conn.commit()
        except Exception as e:
            print(f"插入yolo（base64）失败：{e}")
            return jsonify({"error": f"数据库插入失败：{e}"}), 500

        # 更新任务状态
        with _task_lock:
            _task_states[task_id] = {
                "status": "uploaded",
                "logs": [],
                "img_obb": None,
                "img_dial": None,
                "img_fitting": None,
                "detect_time": None,
                "serial_number": None,
                "reading_before": None,
                "reading_after": None,
                "image_path": save_path,
                "image_name": task_id,
                "image_name_all": save_name,
                "orig_filename": "camera_capture.jpg",
                "operator": username,
            }

        return jsonify({
            "message": "上传成功",
            "task_id": task_id,
            "image_url": f"/image/{save_path}",
            "filename": "camera_capture.jpg"
        })

    except Exception as e:
        return jsonify({"error": f"上传失败：{str(e)}"}), 500
    
    finally:
        if conn:
            conn.close()

#  1. 开始检测 
@app.route("/api/wechat/detect", methods=["POST"])
@require_token
def wechat_detect():
    """微信小程序:开始检测"""
    data = get_json()
    task_id = data.get("task_id")
    
    if not task_id or task_id not in _task_states:
        return jsonify({"error": "task_id无效"}), 400

    # 从 JWT Token 获取用户ID(不是 session)
    user_id = request.current_user.get('user_id')
    if not user_id:
        return jsonify({"error": "用户认证失败"}), 401

    st = _task_states[task_id]
    if st["status"] == "running":
        return jsonify({"error": "检测已在进行中"}), 400

    with _task_lock:
        st["status"] = "pending"
        st["logs"] = ["开始检测..."]

    # 启动检测线程
    threading.Thread(
        target=_run_detection,
        args=(
            task_id,
            st["image_path"],
            st["image_name"],
            st["image_name_all"],
            user_id,
        ),
        daemon=True
    ).start()

    return jsonify({
        "task_id": task_id,
        "message": "检测已启动"
    })


#  2. 轮询检测状态 
@app.route("/api/wechat/poll/<task_id>", methods=["GET"])
@require_token
def wechat_poll(task_id):
    """微信小程序:轮询检测状态"""
    if task_id not in _task_states:
        return jsonify({"error": "task_id不存在"}), 404
    
    with _task_lock:
        st = dict(_task_states[task_id])
    
    # 图片路径转换函数
    img = lambda p: f"/image/{_fwdpath(p)}" if p else None
    
    return jsonify({
        "status": st["status"],
        "logs": st.get("logs", []),
        "img_obb": img(st.get("img_obb")),
        "img_dial": img(st.get("img_dial")),
        "img_fitting": img(st.get("img_fitting")),
        "detect_time": st.get("detect_time"),
        "serial_number": st.get("serial_number"),
        "reading_before": st.get("reading_before"),
        "reading_after": st.get("reading_after"),
        "error": st.get("error"),
    })


#  3. 确认结果 
@app.route("/api/wechat/confirm", methods=["POST"])
@require_token
def wechat_confirm():
    """微信小程序:确认检测结果"""
    data = get_json()
    task_id = data.get("task_id")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if task_id:
        db_update_yolo(task_id, is_confirmed=1, confirmed_at=now)
    
    # 从 JWT Token 获取用户名
    username = request.current_user.get('username', 'unknown')
    
    # 记录日志
    with open(TXT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{username}][{task_id}] confirmed. {now}\n")
    
    return jsonify({"message": "已确认"})


#  4. 修改读数 
# @app.route("/api/wechat/modify", methods=["POST"])
# @require_token
# def wechat_modify():
#     """微信小程序:修改读数"""
#     data = get_json()
#     value = data.get("value", "").strip()
#     task_id = data.get("task_id")
    
#     if not value:
#         return jsonify({"error": "修改值不能为空"}), 400
    
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     if task_id:
#         try:
#             db_update_yolo(task_id, reading_after=float(value))
#         except (ValueError, TypeError):
#             pass  # 输入不是数字就不更新
    
#     # 从 JWT Token 获取用户名
#     username = request.current_user.get('username', 'unknown')
    
#     # 记录日志
#     with open(TXT_LOG_PATH, "a", encoding="utf-8") as f:
#         f.write(f"[{username}][{task_id}] corrected:{value} {now}\n")
    
#     return jsonify({
#         "message": "已修改",
#         "value": value
#     })
@app.route("/api/wechat/modify", methods=["POST"])
@require_token
def wechat_modify():
    """微信小程序: 修改读数"""

    data = request.get_json()

    value = data.get("value", None)
    task_id = data.get("task_id")

    # ✅ 判空（兼容 0）
    if value is None:
        return jsonify({"error": "修改值不能为空"}), 400

    # ✅ 强制转 float（核心修复）
    try:
        value = float(value)
    except (ValueError, TypeError):
        return jsonify({"error": "请输入有效数字"}), 400

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ✅ 更新数据库
    if task_id:
        try:
            db_update_yolo(task_id, reading_after=value)
        except Exception as e:
            return jsonify({"error": f"数据库更新失败: {e}"}), 500

    # ✅ 从 JWT 获取用户名
    username = request.current_user.get('username', 'unknown')

    # ✅ 记录日志（注意这里要转字符串）
    try:
        with open(TXT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{username}][{task_id}] corrected:{value} {now}\n")
    except Exception as e:
        print("日志写入失败:", e)

    return jsonify({
        "message": "已修改",
        "value": value
    })

#  5. 清除日志 
@app.route("/api/wechat/clear", methods=["POST"])
@require_token
def wechat_clear():
    """微信小程序:清除日志"""
    with open(TXT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")
    
    return jsonify({"message": "日志已清除"})


#  6. 获取日志 
@app.route("/api/wechat/get_log", methods=["GET"])
@require_token
def wechat_get_log():
    """微信小程序:获取日志"""
    try:
        with open(TXT_LOG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    
    return jsonify({"content": content})


#  7. 获取历史记录 
@app.route("/api/wechat/history", methods=["GET"])
@require_token
def wechat_history():
    """微信小程序:获取历史记录"""
    # 从 JWT Token 获取用户ID
    user_id = request.current_user.get('user_id')
    
    # 分页参数
    page = max(1, int(request.args.get("page", 1)))
    size = min(50, int(request.args.get("size", 20)))
    offset = (page - 1) * size
    
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                # 查询用户的历史记录
                cur.execute(
                    "SELECT task_id, serial_number, reading_before, reading_after, "
                    "detect_status, is_confirmed, detected_at, created_at "
                    "FROM `yolo` WHERE user_id=%s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (user_id, size, offset)
                )
                rows = cur.fetchall()
                
                # 查询总数
                cur.execute(
                    "SELECT COUNT(*) AS total FROM `yolo` WHERE user_id=%s",
                    (user_id,)
                )
                total = cur.fetchone()["total"]
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # datetime 不能直接 JSON 序列化,转成字符串
    records = []
    for r in rows:
        records.append({
            "task_id": r["task_id"],
            "serial_number": r["serial_number"],
            "reading_before": float(r["reading_before"]) if r["reading_before"] is not None else None,
            "reading_after": float(r["reading_after"]) if r["reading_after"] is not None else None,
            "detect_status": r["detect_status"],
            "is_confirmed": bool(r["is_confirmed"]),
            "detected_at": str(r["detected_at"]) if r["detected_at"] else None,
            "created_at": str(r["created_at"]) if r["created_at"] else None,
        })
    
    return jsonify({
        "total": total,
        "page": page,
        "size": size,
        "records": records
    })


#  8. 获取序列号历史 
@app.route("/api/wechat/serial_history", methods=["GET"])
@require_token
def wechat_serial_history():
    """微信小程序:获取指定序列号的历史记录"""
    serial = request.args.get("serial", "").strip()
    limit = min(int(request.args.get("limit", 60)), 200)
    
    if not serial:
        return jsonify({"error": "serial参数不能为空"}), 400
    
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT task_id, reading_before, reading_after, detected_at, is_confirmed "
                    "FROM `yolo` WHERE serial_number=%s AND detect_status='success' "
                    "AND detected_at IS NOT NULL ORDER BY detected_at ASC LIMIT %s",
                    (serial, limit)
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    records = []
    for r in rows:
        records.append({
            "task_id": r["task_id"],
            "reading_before": float(r["reading_before"]) if r["reading_before"] is not None else None,
            "reading_after": float(r["reading_after"]) if r["reading_after"] is not None else None,
            "detected_at": str(r["detected_at"]) if r["detected_at"] else None,
            "is_confirmed": bool(r["is_confirmed"]),
        })
    
    return jsonify({
        "serial": serial,
        "count": len(records),
        "records": records
    })

# 启动  
if __name__ == "__main__":
    init_db()

    print("="*55)
    print("🚀 服务启动成功")
    print("👉 Web访问地址：")
    print(NGROK_URL)

    print("="*55)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True
    )

# if __name__ == "__main__":
#     init_db()

#     # 检测有没有装pyOpenSSL，有的话启用HTTPS
#     # 摄像头API（getUserMedia）在非localhost的http下不能用，需要HTTPS
#     # pip install pyOpenSSL 就行
#     # 👇 固定写法
#     ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
#     ssl_ctx.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

#     try:
#         import OpenSSL  # noqa
#         ssl_ctx = 'adhoc'
#         print("="*55)
#         print("检测到pyOpenSSL，启用HTTPS自签名证书")
#         print("本机：https://127.0.0.1:5000")
#         print("局域网：https://<本机IP>:5000")
#         print("浏览器会提示证书不安全，点高级->继续访问就行")
#         print("="*55)
#     except ImportError:
#         print("="*55)
#         print("没有pyOpenSSL，HTTP模式启动")
#         print("摄像头只在 http://127.0.0.1:5000 下可用")
#         print("局域网要用摄像头的话：pip install pyOpenSSL 然后重启")
#         print("="*55)

#     # app.run(
#     #     host="0.0.0.0", 
#     #     port=5000, 
#     #     debug=True, 
#     #     threaded=True,
#     #     # ssl_context=ssl_ctx
#     #     ssl_context=('cert.pem', 'key.pem')  # 用真实证书
#     # )
#     app.run(
#         host="0.0.0.0",
#         port=5000,
#         debug=True,
#         threaded=True
#     )