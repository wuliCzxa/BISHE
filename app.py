"""
app.py — 指针式仪表读数检测识别系统（Flask + MySQL 版）
========================================================
安装依赖（无需 bcrypt，已用 werkzeug 内置替代）：
    pip install flask pymysql pandas openpyxl ultralytics opencv-python numpy

启动步骤：
    1. 修改下方 DB_CONFIG 中的 password（及 host/port/user 如有不同）
    2. 确保 MySQL 已启动，数据库 BISHE 可由 init_db() 自动创建
    3. python app.py
    4. 浏览器访问 http://localhost:5000/login
       默认账号：admin / admin123
"""

import os, math, datetime, threading, traceback
from math import sqrt
from functools import wraps

import cv2
import numpy as np
import pandas as pd
import pymysql
import pymysql.cursors
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (Flask, render_template, request, jsonify,
                   send_file, abort, redirect, url_for, session)
from ultralytics import YOLO

# ── 从 config.py 读取配置（找不到时使用内置默认值）──────────
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
    print("[CONFIG] ✔ 已从 config.py 加载配置")
except (ImportError, AttributeError) as _ce:
    print(f"[CONFIG] ⚠ config.py 加载失败（{_ce}），使用内置默认值")
    _DB_HOST="localhost"; _DB_PORT=3306; _DB_USER="root"; _DB_PASSWORD=""
    _DB_NAME="BISHE"
    FILE_EXCEL_PATH="../ultralytics001/yolo_obb/序号标记对照表.xlsx"
    MODEL_PATH1="../ultralytics001/yolo_obb/weight/1biaopan_all/weights/best.pt"
    MODEL_PATH2="../ultralytics001/yolo_obb/weight/2biaopan_nolabel/weights/best.pt"
    MODEL_PATH3="../ultralytics001/yolo_obb/weight/3biaopan_label/weights/best.pt"
    MODEL_PATH4="../ultralytics001/yolo_obb/weight/4read/weights/best.pt"
    TXT_LOG_PATH="../ultralytics001/yolo_obb/Result_pointer.txt"
    UPLOAD_FOLDER="uploads"; OUTPUT_FOLDER="outputs"
    _SECRET_KEY="bishe_flask_secret_2025"

# ══════════════════════════════════════════════════════════════
#  Flask 初始化
# ══════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = _SECRET_KEY

# ══════════════════════════════════════════════════════════════
#  MySQL 连接
# ══════════════════════════════════════════════════════════════
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

def get_db() -> pymysql.connections.Connection:
    """获取 MySQL 连接，失败时抛出 RuntimeError（包含可读错误信息）"""
    try:
        return pymysql.connect(**DB_CONFIG)
    except pymysql.err.OperationalError as e:
        code = e.args[0] if e.args else 0
        if code == 1045:
            hint = f"密码错误，请修改 config.py 中的 DB_PASSWORD（当前值：'{_DB_PASSWORD}'）"
        elif code in (2003, 2002):
            hint = (f"MySQL 服务未启动或端口 {_DB_PORT} 无法连接。"
                    "请先运行 python check_db.py 查看详细修复步骤。")
        else:
            hint = str(e)
        raise RuntimeError(f"MySQL 连接失败：{hint}") from e

# ══════════════════════════════════════════════════════════════
#  全局 JSON 错误处理（确保 /api/* 始终返回 JSON，不返回 HTML）
# ══════════════════════════════════════════════════════════════
@app.errorhandler(Exception)
def handle_any_exception(e):
    tb = traceback.format_exc()
    print(f"[UNHANDLED] {tb}")
    # API 路由 / POST 请求 → 返回 JSON
    if request.path.startswith("/api/") or request.method == "POST":
        msg = str(e)
        # 对用户隐藏内部细节，但保留关键提示
        if "MySQL" in msg or "Can't connect" in msg or "Connection refused" in msg:
            msg = "数据库连接失败，请检查 MySQL 是否启动及 DB_CONFIG 密码是否正确"
        return jsonify({"error": msg}), 500
    return f"<pre>{tb}</pre>", 500

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "接口不存在"}), 404
    return redirect(url_for("login_page"))

# 路径均已从 config.py 加载，此处仅确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  密码工具（werkzeug 内置，无需额外安装）
# ══════════════════════════════════════════════════════════════
def _hash_pw(plain: str) -> str:
    return generate_password_hash(plain)

def _check_pw(plain: str, hashed: str) -> bool:
    return check_password_hash(hashed, plain)

# ══════════════════════════════════════════════════════════════
#  数据库初始化（自动建库建表 + 写入默认管理员）
# ══════════════════════════════════════════════════════════════
def init_db():
    print("[DB] 正在初始化数据库...")
    # ── 第一步：不指定 database，先确保 BISHE 库存在 ──
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
        print(f"[DB] ⚠ 无法创建数据库：{e}")
        print("[DB] 请手动执行：CREATE DATABASE BISHE DEFAULT CHARSET utf8mb4;")
        return

    # ── 第二步：连接到 BISHE，建表 ──
    try:
        conn = get_db()
    except RuntimeError as e:
        print(f"[DB] ⚠ {e}")
        return

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
            # 默认管理员（admin / admin123）
            cur.execute("SELECT id FROM `user` WHERE username='admin' LIMIT 1")
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO `user`(username,password,user_level) VALUES(%s,%s,%s)",
                    ("admin", _hash_pw("admin123"), "super_admin")
                )
                print("[DB] ✔ 已创建默认账号 admin / admin123")
        conn.commit()
        print("[DB] ✔ 数据库初始化完成")
    except Exception as e:
        print(f"[DB] ⚠ 建表失败：{e}")
    finally:
        conn.close()

# ══════════════════════════════════════════════════════════════
#  序号-表计对照字典
# ══════════════════════════════════════════════════════════════
try:
    _df = pd.read_excel(FILE_EXCEL_PATH, engine="openpyxl")
    _df["序号"] = _df["序号"].astype(str)
    MY_DICT: dict = pd.Series(_df["表计"].values, index=_df["序号"]).to_dict()
    print(f"[DICT] ✔ 序号对照表加载 {len(MY_DICT)} 条")
except Exception as e:
    print(f"[DICT] ⚠ 序号标记对照表加载失败：{e}")
    MY_DICT = {}

# ══════════════════════════════════════════════════════════════
#  task_id 生成：YYYY-MM-DD-N
# ══════════════════════════════════════════════════════════════
_date_counter: dict = {}
_counter_lock = threading.Lock()

def generate_task_id() -> str:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    with _counter_lock:
        _date_counter[today] = _date_counter.get(today, 0) + 1
        n = _date_counter[today]
    return f"{today}-{n}"

# ══════════════════════════════════════════════════════════════
#  内存任务状态
# ══════════════════════════════════════════════════════════════
_task_states: dict = {}
_task_lock = threading.Lock()

def _fwdpath(p: str) -> str:
    """
    将路径中的反斜杠统一替换为正斜杠，确保 Windows 下路径
    可以安全嵌入 URL（如 /image/outputs/...）。
    """
    # return p.replace('\\', '/').replace('\\', '/').replace('\', '/') if p else p
    return p.replace('\\', '/').replace('\\', '/')  if p else p
# ══════════════════════════════════════════════════════════════
#  数学工具函数
# ══════════════════════════════════════════════════════════════
def calc_intersection(p1, p2, p3, p4):
    try:
        m1=(p2[1]-p1[1])/(p2[0]-p1[0]); b1=p1[1]-m1*p1[0]
        m2=(p4[1]-p3[1])/(p4[0]-p3[0]); b2=p3[1]-m2*p3[0]
        if m1==m2: return None
        x=(b2-b1)/(m1-m2); return (x, m1*x+b1)
    except ZeroDivisionError: return None

def dist(a, b):
    return int(sqrt((a[0]-b[0])**2+(a[1]-b[1])**2))

def clock_angle(v1, v2):
    n=np.linalg.norm(v1)*np.linalg.norm(v2)
    rho=np.rad2deg(np.arcsin(np.clip(np.cross(v1,v2)/n,-1,1)))
    theta=np.rad2deg(np.arccos(np.clip(np.dot(v1,v2)/n,-1,1)))
    return theta if rho>0 else 360-theta

def mid_point(x1,y1,x2,y2,x3,y3,x4,y4):
    edges=[((x1,y1),(x2,y2)),((x2,y2),(x3,y3)),((x3,y3),(x4,y4)),((x4,y4),(x1,y1))]
    el=[(e,math.hypot(e[1][0]-e[0][0],e[1][1]-e[0][1])) for e in edges]
    el.sort(key=lambda x:x[1])
    mid=lambda p,q:((p[0]+q[0])/2,(p[1]+q[1])/2)
    return mid(*el[0][0]), mid(*el[1][0])

# ══════════════════════════════════════════════════════════════
#  DB 辅助：安全更新 yolo 记录
# ══════════════════════════════════════════════════════════════
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
        print(f"[DB] yolo 更新失败 task_id={task_id}：{e}")

# ══════════════════════════════════════════════════════════════
#  核心检测流程（后台线程）
# ══════════════════════════════════════════════════════════════
def _run_detection(task_id, image_path, image_name, image_name_all):
    def log(msg):
        with _task_lock:
            _task_states[task_id]["logs"].append(msg)

    state = _task_states[task_id]
    state["status"] = "running"
    db_update_yolo(task_id, detect_status="running")

    try:
        rdir = os.path.join(OUTPUT_FOLDER,
               f"outputs-{datetime.datetime.now().strftime('%Y-%m-%d')}")
        rimg = os.path.join(rdir, image_name)
        os.makedirs(rimg, exist_ok=True)
        txt1 = os.path.join(rimg, "result.txt")

        log("▶ 步骤 1 / 5：裁剪仪表盘区域...")
        r1   = YOLO(MODEL_PATH1)(source=image_path, save=True, save_txt=True, save_crop=True, conf=0.7)
        sp1  = str(r1[0].save_dir)
        i1   = cv2.imread(os.path.join(sp1,"crops","Instrument",image_name_all))
        obb_path = os.path.join(rimg, image_name+"_all.jpg")
        cv2.imwrite(obb_path, i1)
        state["img_obb"] = _fwdpath(obb_path)
        db_update_yolo(task_id, obb_img_path=_fwdpath(obb_path))
        log("✔ 步骤 1 完成：仪表盘（带标签）已裁剪")
        imgp = obb_path

        log("▶ 步骤 2 / 5：裁剪表盘（无标签）...")
        r2   = YOLO(MODEL_PATH2)(source=imgp, save=True, save_txt=True, save_crop=True, conf=0.7)
        sp2  = str(r2[0].save_dir)
        i2   = cv2.imread(os.path.join(sp2,"crops","Pointer",image_name+"_all.jpg"))
        dial_path = os.path.join(rimg, image_name+"_biaopan.jpg")
        cv2.imwrite(dial_path, i2)
        state["img_dial"] = _fwdpath(dial_path)
        db_update_yolo(task_id, dial_img_path=_fwdpath(dial_path))
        log("✔ 步骤 2 完成：表盘图已保存")

        log("▶ 步骤 3 / 5：裁剪序号标签区域...")
        r3   = YOLO(MODEL_PATH3)(source=imgp, save=True, save_txt=True, save_crop=True, conf=0.7)
        sp3  = str(r3[0].save_dir)
        i3   = cv2.imread(os.path.join(sp3,"crops","Label",image_name+"_all.jpg"))
        lbl_path = os.path.join(rimg, image_name+"_biaoqian.jpg")
        cv2.imwrite(lbl_path, i3)
        db_update_yolo(task_id, label_img_path=lbl_path)
        log("✔ 步骤 3 完成：标签区域图已保存")

        log("▶ 步骤 4 / 5：识别序号标签...")
        r4   = YOLO(MODEL_PATH4)(source=lbl_path, save=True, save_txt=True, conf=0.7)
        lp4  = os.path.join(str(r4[0].save_dir), "labels")
        log("✔ 步骤 4 完成：序号识别完毕")

        log("▶ 步骤 5 / 5：OBB 关键点检测与读数计算...")
        r5   = YOLO(MODEL_PATH4)(source=dial_path, save=True, save_txt=True, conf=0.7)
        sp5  = str(r5[0].save_dir)
        lp5  = os.path.join(sp5, "labels")

        image = cv2.imread(os.path.join(sp5, image_name+"_biaopan.jpg"))
        h, w  = image.shape[:2]

        rows = []
        with open(os.path.join(lp5, image_name+"_biaopan.txt")) as f:
            for line in f: rows.append(line.strip().split())
        sr = sorted(rows, key=lambda x:(float(x[0]),float(x[1])))

        def rc(r,i): return float(r[i])
        xs1,ys1=w*rc(sr[0],7),h*rc(sr[0],8); xs2,ys2=w*rc(sr[0],1),h*rc(sr[0],2)
        xs3,ys3=w*rc(sr[0],3),h*rc(sr[0],4); xs4,ys4=w*rc(sr[0],5),h*rc(sr[0],6)
        xsf,ysf=(xs1+xs2+xs3+xs4)/4,(ys1+ys2+ys3+ys4)/4

        xe1,ye1=w*rc(sr[1],1),h*rc(sr[1],2); xe2,ye2=w*rc(sr[1],3),h*rc(sr[1],4)
        xe3,ye3=w*rc(sr[1],5),h*rc(sr[1],6); xe4,ye4=w*rc(sr[1],7),h*rc(sr[1],8)
        xef,yef=(xe1+xe2+xe3+xe4)/4,(ye1+ye2+ye3+ye4)/4

        xp1,yp1=w*rc(sr[3],7),h*rc(sr[3],8); xp2,yp2=w*rc(sr[3],1),h*rc(sr[3],2)
        xp3,yp3=w*rc(sr[3],3),h*rc(sr[3],4); xp4,yp4=w*rc(sr[3],5),h*rc(sr[3],6)
        xpf,ypf=(xp1+xp2+xp3+xp4)/4,(yp1+yp2+yp3+yp4)/4

        xz1,yz1=w*rc(sr[2],1),h*rc(sr[2],2); xz2,yz2=w*rc(sr[2],3),h*rc(sr[2],4)
        xz3,yz3=w*rc(sr[2],5),h*rc(sr[2],6); xz4,yz4=w*rc(sr[2],7),h*rc(sr[2],8)
        xzf,yzf=(xz1+xz2+xz3+xz4)/4,(yz1+yz2+yz3+yz4)/4

        (ss1,ss2),(ss3,ss4)=mid_point(xs1,ys1,xs2,ys2,xs3,ys3,xs4,ys4)
        (ee1,ee2),(ee3,ee4)=mid_point(xe1,ye1,xe2,ye2,xe3,ye3,xe4,ye4)
        (pp1,pp2),(pp3,pp4)=mid_point(xp1,yp1,xp2,yp2,xp3,yp3,xp4,yp4)
        (zz1,zz2),(zz3,zz4)=mid_point(xz1,yz1,xz2,yz2,xz3,yz3,xz4,yz4)
        p1=(ss1,ss2);p2=(ss3,ss4);p3=(ee1,ee2);p4=(ee3,ee4)
        p5=(pp1,pp2);p6=(pp3,pp4);p7=(zz1,zz2);p8=(zz3,zz4)

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

        for pt,col in [((int(ise2[0]),int(ise2[1])),(0,0,255)),((int(isp[0]),int(isp[1])),(0,0,120)),
                       ((int(iep[0]),int(iep[1])),(0,0,0)),((int(cx),int(cy)),(0,0,255)),
                       ((int(xzf),int(yzf)),(0,0,255)),((int(cxx),int(cyy)),(0,0,255))]:
            cv2.circle(image,pt,5,col,-1)

        # ── 保存拟合图（含完整校验）──────────────────────────────
        fit_path = os.path.join(rimg, image_name + "_fitting.jpg")
        write_ok = cv2.imwrite(fit_path, image)
        if not write_ok:
            # cv2.imwrite 可能因路径问题返回 False，尝试用绝对路径重写
            fit_path_abs = os.path.abspath(fit_path)
            write_ok = cv2.imwrite(fit_path_abs, image)
            if write_ok:
                fit_path = fit_path_abs
                print(f"[IMG] 绝对路径写入成功：{fit_path}")
            else:
                print(f"[IMG] ⚠ 拟合图保存失败，路径：{fit_path}")

        if write_ok and os.path.isfile(fit_path) and os.path.getsize(fit_path) > 0:
            # ★ Fix: 用 _fwdpath 确保路径使用正斜杠（解决 Windows 反斜杠问题）
            state["img_fitting"] = _fwdpath(fit_path)
            print(f"[IMG] ✔ 拟合图已保存：{state['img_fitting']}")
        else:
            print(f"[IMG] ✘ 拟合图文件不存在或为空：{fit_path}")

        rv1=clock_angle([xsf-cx,ysf-cy],[xpf-cx,ypf-cy])/clock_angle([xsf-cx,ysf-cy],[xef-cx,yef-cy])-0.1
        acx=(cx+cxx)/2; acy=(cy+cyy)/2
        th3=clock_angle([xsf-acx,ysf-acy],[xpf-acx,ypf-acy])
        th4=clock_angle([xsf-acx,ysf-acy],[xef-acx,yef-acy])
        rv2=th3/th4-0.1
        rv3=clock_angle([xsf-acx,ysf-acy],[xzf-acx,yzf-acy])/th4-0.1
        final=rv2+(0.4-rv3)/2

        snum="未知"
        bq_txt=os.path.join(lp4, image_name+"_biaoqian.txt")
        with open(bq_txt) as f:
            for line in f:
                snum=MY_DICT.get(line.split()[0],f"序号{line.split()[0]}")
                break

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
        log(f"✔ 步骤 5 完成 | 序号：{snum}")
        log(f"📌 修正前读数：{rv1:.6f}")
        log(f"📌 修正后读数：{final:.6f}")

    except Exception as exc:
        traceback.print_exc()
        with _task_lock:
            _task_states[task_id]["status"]="error"
            _task_states[task_id]["error"]=str(exc)
        db_update_yolo(task_id, detect_status="failed")
        log(f"❌ 检测出错：{exc}")

# ══════════════════════════════════════════════════════════════
#  登录鉴权装饰器
# ══════════════════════════════════════════════════════════════
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            if request.method == "GET":
                return redirect(url_for("login_page"))
            return jsonify({"error":"请先登录","redirect":"/login"}), 401
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════════════════════════
#  辅助：安全解析请求 JSON
# ══════════════════════════════════════════════════════════════
def get_json() -> dict:
    """安全获取请求 JSON，失败返回空字典"""
    return request.get_json(force=True, silent=True) or {}

# ══════════════════════════════════════════════════════════════
#  路由 — 认证
# ══════════════════════════════════════════════════════════════
@app.route("/login")
def login_page():
    if "username" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    data     = get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    # ── 查询数据库 ──
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
        return jsonify({"error": f"数据库查询失败：{e}"}), 500
    finally:
        conn.close()

    if not user or not _check_pw(password, user["password"]):
        return jsonify({"error": "用户名或密码错误"}), 401

    session["username"]   = user["username"]
    session["user_id"]    = user["id"]
    session["user_level"] = user["user_level"]
    return jsonify({
        "message":    "登录成功",
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
        return jsonify({"error": "用户名至少 3 个字符"}), 400
    if len(password) < 6:
        return jsonify({"error": "密码至少 6 个字符"}), 400
    if user_level not in ("super_admin", "admin", "user"):
        user_level = "user"

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

    return jsonify({"message": "注册成功，请登录"})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# ══════════════════════════════════════════════════════════════
#  路由 — 主业务
# ══════════════════════════════════════════════════════════════
@app.route("/")
@login_required
def index():
    return render_template("index.html",
                           username   = session.get("username"),
                           user_level = session.get("user_level"))


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "未收到文件"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    task_id   = generate_task_id()
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
        print(f"[DB] 插入 yolo 失败：{e}")

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
    """
    接收前端摄像头拍摄的 base64 图片（data:image/jpeg;base64,... 或纯 base64）。
    与 /upload 逻辑完全一致，差异仅在于图片来源为 base64 字符串。
    """
    import base64 as _b64
    data   = get_json()
    b64str = data.get("image", "").strip()
    if not b64str:
        return jsonify({"error": "未收到图片数据"}), 400

    # 去掉 data URL 前缀（如 "data:image/jpeg;base64,"）
    if "," in b64str:
        b64str = b64str.split(",", 1)[1]

    try:
        img_bytes = _b64.b64decode(b64str)
    except Exception:
        return jsonify({"error": "base64 解码失败，图片数据格式有误"}), 400

    task_id   = generate_task_id()
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
        print(f"[DB] 插入 yolo（base64）失败：{e}")

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
        return jsonify({"error": "task_id 无效"}), 400
    st = _task_states[task_id]
    if st["status"] == "running":
        return jsonify({"error": "检测已在进行中"}), 400
    with _task_lock:
        st["status"] = "pending"; st["logs"] = ["🚀 开始检测流程..."]
    threading.Thread(
        target=_run_detection,
        args=(task_id, st["image_path"], st["image_name"], st["image_name_all"]),
        daemon=True
    ).start()
    return jsonify({"task_id": task_id, "message": "检测已启动"})


@app.route("/poll/<task_id>")
@login_required
def poll(task_id):
    if task_id not in _task_states:
        return jsonify({"error": "task_id 不存在"}), 404
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
            pass
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

    # datetime 序列化
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
        return jsonify({"error": "serial 参数不能为空"}), 400
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


@app.route("/image/<path:filepath>")
@login_required
def serve_image(filepath):
    # ★ Fix: 统一替换反斜杠，兼容 Windows 路径
    filepath = filepath.replace('\\', '/').replace('\\', '/').replace('\\', '/')
    # 同时尝试正斜杠和系统分隔符版本
    abs_path = os.path.abspath(filepath)
    # 若路径带正斜杠在 Windows 上 abspath 可能不对，也尝试 os.sep 版本
    alt_path = os.path.abspath(filepath.replace('/', os.sep))
    allowed  = [os.path.abspath(OUTPUT_FOLDER), os.path.abspath(UPLOAD_FOLDER)]

    # 用两个候选路径都尝试
    valid_path = None
    for candidate in [abs_path, alt_path]:
        if (any(candidate.startswith(a) for a in allowed)
                and os.path.isfile(candidate)):
            valid_path = candidate
            break

    if valid_path is None:
        # 详细日志帮助排查路径问题
        print(f"[SERVE] ⚠ 路径不可访问：{filepath!r}")
        print(f"[SERVE]   abs_path={abs_path!r}")
        print(f"[SERVE]   allowed={[str(a) for a in allowed]}")
        abort(404)

    return send_file(valid_path)


# ══════════════════════════════════════════════════════════════
#  启动
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_db()

    # ★ 修复：自动检测 pyOpenSSL，有则启用 HTTPS。
    #   浏览器的摄像头 API (getUserMedia) 只在安全上下文（HTTPS / localhost）下可用。
    #   启用 HTTPS 后，局域网 IP（172.x / 192.168.x 等）也能正常调用摄像头。
    #
    #   安装依赖：pip install pyOpenSSL
    #   启用后访问：https://<本机IP>:5000  （浏览器提示"不安全"时点击"高级"→"继续访问"）
    ssl_ctx = None
    try:
        import OpenSSL  # noqa: F401  — 仅检测是否已安装
        ssl_ctx = 'adhoc'
        print("=" * 60)
        print("[SSL] ✔ 检测到 pyOpenSSL，已启用 HTTPS 自签名证书")
        print("[SSL]   本机访问：  https://127.0.0.1:5000")
        print("[SSL]   局域网访问：https://<本机IP>:5000")
        print("[SSL]   ⚠ 首次访问时浏览器会提示证书不受信任，")
        print("[SSL]     点击「高级」→「继续访问（不安全）」即可正常使用摄像头。")
        print("=" * 60)
    except ImportError:
        print("=" * 60)
        print("[SSL] ⚠ 未检测到 pyOpenSSL，以 HTTP 模式启动。")
        print("[SSL]   摄像头功能仅在 http://127.0.0.1:5000（本机）下可用。")
        print("[SSL]   如需在局域网 IP 下也使用摄像头，请执行：")
        print("[SSL]       pip install pyOpenSSL")
        print("[SSL]   然后重启本程序。")
        print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True,
            ssl_context=ssl_ctx)
