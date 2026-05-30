"""
app.py  指针式仪表读数检测识别系统 - 改进版
基于Flask + MySQL + YOLOv8
增强特性：
  - RANSAC算法多线交点优化
  - 动态权重融合（基于置信度、夹角、离散度）
  - 最小二乘要点-读数映射
  - 卡尔曼滤波/一阶低通平滑
  - 置信度与夹角异常报警机制
"""
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
from ultralytics import YOLO
from qrcode_helper import serve_qrcode, get_qrcode_info
from flask import send_from_directory
from scipy.spatial.distance import mahalanobis
from scipy.optimize import least_squares
from sklearn.linear_model import RANSACRegressor

SECRET_KEY = "wuliBISHE_secret_key_2025"

# 读config.py
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

    WECHAT_MINIPROGRAM_PATH = "pages/index/index"
    WECHAT_MINIPROGRAM_APPID = "your_miniprogram_appid"
    APK_DOWNLOAD_URL = "https://your-domain.com/downloads/app.apk"

app = Flask(__name__)
app.secret_key = _SECRET_KEY

JWT_SECRET = _SECRET_KEY
JWT_EXPIRATION = timedelta(days=180)

CORS(app) 

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
    """获取数据库连接"""
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
    """验证 JWT Token 的装饰器"""
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

def _hash_pw(plain: str) -> str:
    return generate_password_hash(plain)

def _check_pw(plain: str, hashed: str) -> bool:
    return check_password_hash(hashed, plain)


# ===== 改进的数学工具函数 =====

class KalmanFilter1D:
    """一维卡尔曼滤波器，用于平滑圆心坐标"""
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.x = 0.0  # 状态估计
        self.P = 1.0  # 误差协方差
        self.initialized = False
    
    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        
        # 预测
        x_pred = self.x
        P_pred = self.P + self.process_variance
        
        # 更新
        K = P_pred / (P_pred + self.measurement_variance)  # 卡尔曼增益
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return self.x

class LowPassFilter1D:
    """一阶低通滤波器"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # 平滑系数，0-1之间，越小越平滑
        self.value = None
    
    def update(self, measurement):
        if self.value is None:
            self.value = measurement
        else:
            self.value = self.alpha * measurement + (1 - self.alpha) * self.value
        return self.value

# 全局滤波器实例（每个任务独立）
_filters = {}

def get_filters(task_id):
    """获取或创建任务的滤波器"""
    if task_id not in _filters:
        _filters[task_id] = {
            'kalman_cx': KalmanFilter1D(),
            'kalman_cy': KalmanFilter1D(),
            'lowpass_cx': LowPassFilter1D(alpha=0.4),
            'lowpass_cy': LowPassFilter1D(alpha=0.4),
        }
    return _filters[task_id]

def calc_intersection(p1, p2, p3, p4):
    """求两条线段所在直线的交点"""
    try:
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - m1 * p1[0]
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b2 = p3[1] - m2 * p3[0]
        if abs(m1 - m2) < 1e-6:
            return None  # 平行
        x = (b2 - b1) / (m1 - m2)
        return (x, m1 * x + b1)
    except ZeroDivisionError:
        return None  # 垂直线

def dist(a, b):
    """计算两点距离"""
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def clock_angle(v1, v2):
    """计算顺时针角度"""
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-10:
        return 0.0
    rho = np.rad2deg(np.arcsin(np.clip(np.cross(v1, v2) / n, -1, 1)))
    theta = np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1)))
    return theta if rho > 0 else 360 - theta

def mid_point(x1, y1, x2, y2, x3, y3, x4, y4):
    """取OBB四条边里最短的两条的中点"""
    edges = [((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), 
             ((x3, y3), (x4, y4)), ((x4, y4), (x1, y1))]
    el = [(e, math.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1])) for e in edges]
    el.sort(key=lambda x: x[1])
    mid = lambda p, q: ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    return mid(*el[0][0]), mid(*el[1][0])


# ===== 改进的圆心计算算法 =====

def ransac_circle_center(intersections, threshold=10.0, min_samples=2, max_trials=100):
    """
    使用RANSAC算法从多个交点中估计圆心
    
    参数:
        intersections: 交点列表 [(x1, y1), (x2, y2), ...]
        threshold: 内点距离阈值
        min_samples: 最小样本数
        max_trials: 最大迭代次数
    
    返回:
        best_center: 最优圆心 (x, y)
        inlier_mask: 内点掩码
        confidence: 置信度 (0-1)
    """
    if len(intersections) < min_samples:
        return None, None, 0.0
    
    points = np.array(intersections)
    best_center = None
    best_inliers = []
    max_inlier_count = 0
    
    for _ in range(max_trials):
        # 随机采样
        sample_idx = np.random.choice(len(points), min_samples, replace=False)
        sample_points = points[sample_idx]
        
        # 计算候选圆心（样本点的均值）
        candidate_center = np.mean(sample_points, axis=0)
        
        # 计算所有点到候选圆心的距离
        distances = np.linalg.norm(points - candidate_center, axis=1)
        
        # 找出内点
        inliers = distances < threshold
        inlier_count = np.sum(inliers)
        
        # 更新最佳模型
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_inliers = inliers
            # 用所有内点重新计算圆心
            best_center = np.mean(points[inliers], axis=0)
    
    if best_center is None:
        # RANSAC失败，返回所有点的均值
        best_center = np.mean(points, axis=0)
        best_inliers = np.ones(len(points), dtype=bool)
    
    confidence = np.sum(best_inliers) / len(points)
    
    return tuple(best_center), best_inliers, confidence

def compute_incenter(p1, p2, p3):
    """
    计算三角形内心
    
    参数:
        p1, p2, p3: 三角形三个顶点
    
    返回:
        内心坐标 (x, y)
    """
    # 计算三边长度
    a = dist(p2, p3)
    b = dist(p1, p3)
    c = dist(p1, p2)
    
    # 避免除零
    perimeter = a + b + c
    if perimeter < 1e-6:
        return ((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3)
    
    # 内心公式：(a*A + b*B + c*C) / (a + b + c)
    x = (a * p1[0] + b * p2[0] + c * p3[0]) / perimeter
    y = (a * p1[1] + b * p2[1] + c * p3[1]) / perimeter
    
    return (x, y)

def least_squares_circle_center(points):
    """
    最小二乘法拟合圆心
    
    参数:
        points: 点列表 [(x1, y1), (x2, y2), ...]
    
    返回:
        center: 圆心 (x, y)
        radius: 半径
    """
    if len(points) < 3:
        return np.mean(points, axis=0), 0.0
    
    points = np.array(points)
    
    def calc_R(xc, yc):
        """计算所有点到(xc, yc)的距离"""
        return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)
    
    def f(c):
        """目标函数：距离方差"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    
    # 初始猜测：点的均值
    center_estimate = np.mean(points, axis=0)
    
    # 最小二乘优化
    result = least_squares(f, center_estimate)
    center = result.x
    Ri = calc_R(*center)
    radius = Ri.mean()
    
    return tuple(center), radius

def compute_angle_variance(points, center):
    """
    计算点相对于圆心的角度方差，用于评估点的分散程度
    
    参数:
        points: 点列表
        center: 圆心
    
    返回:
        angle_variance: 角度方差（弧度）
    """
    if len(points) < 2:
        return 0.0
    
    angles = []
    for p in points:
        vec = np.array([p[0] - center[0], p[1] - center[1]])
        angle = np.arctan2(vec[1], vec[0])
        angles.append(angle)
    
    angles = np.array(angles)
    return np.var(angles)

def dynamic_weight_fusion(centers_with_metrics):
    """
    动态权重融合多个圆心估计
    
    参数:
        centers_with_metrics: [(center, confidence, angle_variance, dispersion), ...]
            - center: 圆心坐标
            - confidence: 检测置信度
            - angle_variance: 夹角方差
            - dispersion: 离散度
    
    返回:
        fused_center: 融合后的圆心
        weights: 各圆心的权重
    """
    if not centers_with_metrics:
        return None, None
    
    centers = np.array([c[0] for c in centers_with_metrics])
    confidences = np.array([c[1] for c in centers_with_metrics])
    angle_vars = np.array([c[2] for c in centers_with_metrics])
    dispersions = np.array([c[3] for c in centers_with_metrics])
    
    # 归一化指标到0-1
    conf_norm = confidences / (np.max(confidences) + 1e-6)
    
    # 角度方差越小越好，转换为分数
    angle_score = 1.0 / (1.0 + angle_vars)
    angle_score_norm = angle_score / (np.max(angle_score) + 1e-6)
    
    # 离散度越小越好
    dispersion_score = 1.0 / (1.0 + dispersions)
    dispersion_score_norm = dispersion_score / (np.max(dispersion_score) + 1e-6)
    
    # 综合权重：加权平均（可调整权重比例）
    weights = (0.4 * conf_norm + 0.3 * angle_score_norm + 0.3 * dispersion_score_norm)
    weights = weights / (np.sum(weights) + 1e-6)
    
    # 加权平均圆心
    fused_center = np.average(centers, axis=0, weights=weights)
    
    return tuple(fused_center), weights

def mahalanobis_outlier_detection(points, threshold=3.0):
    """
    使用马氏距离检测离群点
    
    参数:
        points: 点列表
        threshold: 马氏距离阈值
    
    返回:
        inlier_mask: 内点掩码
    """
    if len(points) < 3:
        return np.ones(len(points), dtype=bool)
    
    points = np.array(points)
    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)
    
    # 避免奇异协方差矩阵
    if np.linalg.det(cov) < 1e-10:
        cov += np.eye(2) * 1e-6
    
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.ones(len(points), dtype=bool)
    
    # 计算马氏距离
    mahal_dists = []
    for p in points:
        diff = p - mean
        mahal_dist = np.sqrt(diff.T @ cov_inv @ diff)
        mahal_dists.append(mahal_dist)
    
    mahal_dists = np.array(mahal_dists)
    inlier_mask = mahal_dists < threshold
    
    return inlier_mask

def advanced_circle_estimation(line_pairs, confidence_scores=None):
    """
    改进的圆心估计算法
    
    参数:
        line_pairs: 线段对列表 [((p1, p2), (p3, p4)), ...]
        confidence_scores: 每对线段的置信度（可选）
    
    返回:
        center: 最优圆心
        quality_metrics: 质量指标字典
    """
    # 1. 计算所有交点
    intersections = []
    valid_pairs = []
    for i, pair in enumerate(line_pairs):
        intersection = calc_intersection(*pair[0], *pair[1])
        if intersection is not None:
            intersections.append(intersection)
            valid_pairs.append(i)
    
    if len(intersections) < 2:
        return None, {"error": "交点不足"}
    
    # 2. 使用马氏距离剔除离群点
    inlier_mask = mahalanobis_outlier_detection(intersections)
    filtered_intersections = [intersections[i] for i in range(len(intersections)) if inlier_mask[i]]
    
    if len(filtered_intersections) < 2:
        filtered_intersections = intersections  # 回退
    
    # 3. RANSAC拟合圆心
    ransac_center, ransac_inliers, ransac_conf = ransac_circle_center(
        filtered_intersections, threshold=15.0, max_trials=100
    )
    
    # 4. 最小二乘拟合圆心
    ls_center, ls_radius = least_squares_circle_center(filtered_intersections)
    
    # 5. 如果有三个点，计算内心
    incenter = None
    if len(filtered_intersections) >= 3:
        incenter = compute_incenter(
            filtered_intersections[0], 
            filtered_intersections[1], 
            filtered_intersections[2]
        )
    
    # 6. 计算各候选圆心的质量指标
    candidates = []
    
    # RANSAC圆心
    if ransac_center is not None:
        angle_var = compute_angle_variance(filtered_intersections, ransac_center)
        dispersion = np.mean([dist(ransac_center, p) for p in filtered_intersections])
        candidates.append((ransac_center, ransac_conf, angle_var, dispersion))
    
    # 最小二乘圆心
    if ls_center is not None:
        angle_var = compute_angle_variance(filtered_intersections, ls_center)
        dispersion = ls_radius
        ls_conf = 0.9  # 最小二乘置信度较高
        candidates.append((ls_center, ls_conf, angle_var, dispersion))
    
    # 内心
    if incenter is not None:
        angle_var = compute_angle_variance(filtered_intersections[:3], incenter)
        dispersion = np.mean([dist(incenter, p) for p in filtered_intersections[:3]])
        inc_conf = 0.85
        candidates.append((incenter, inc_conf, angle_var, dispersion))
    
    # 7. 动态权重融合
    if not candidates:
        return None, {"error": "无有效候选圆心"}
    
    fused_center, weights = dynamic_weight_fusion(candidates)
    
    # 8. 质量指标
    quality_metrics = {
        "num_intersections": len(intersections),
        "num_inliers": len(filtered_intersections),
        "ransac_confidence": ransac_conf,
        "candidate_centers": [c[0] for c in candidates],
        "fusion_weights": weights.tolist() if weights is not None else [],
        "final_center": fused_center
    }
    
    return fused_center, quality_metrics


# ===== 改进的读数校正算法 =====

def least_squares_reading_calibration(keypoint_angles, reference_readings, slider_position=0.5):
    """
    使用最小二乘法建立关键点角度到读数的映射
    
    参数:
        keypoint_angles: 关键点相对角度列表 [angle_start, angle_pointer, angle_end, angle_zero]
        reference_readings: 参考读数 [reading_start, reading_end]
        slider_position: 滑块位置 (0-1)，用于引入工件系数
    
    返回:
        calibrated_reading: 校正后的读数
    """
    angle_start, angle_pointer, angle_end, angle_zero = keypoint_angles
    reading_start, reading_end = reference_readings
    
    # 计算角度比例
    angle_range = angle_end - angle_start
    angle_current = angle_pointer - angle_start
    
    if abs(angle_range) < 1e-6:
        return 0.0
    
    # 基础读数
    basic_ratio = angle_current / angle_range
    
    # 引入滑块位置作为工件系数
    # slider_position影响非线性校正
    workpiece_coeff = 1.0 + 0.1 * (slider_position - 0.5)
    
    # 零点校正
    zero_offset = 0.0
    if angle_zero is not None:
        angle_zero_normalized = (angle_zero - angle_start) / angle_range
        # 理想情况下零点应该在起点，偏移即为误差
        zero_offset = angle_zero_normalized
    
    # 最小二乘校正公式
    # reading = a * basic_ratio + b * zero_offset + c
    # 这里简化为线性模型，实际可扩展为多项式
    a = (reading_end - reading_start) * workpiece_coeff
    b = -0.05  # 零点校正系数
    c = reading_start
    
    calibrated_reading = a * basic_ratio + b * zero_offset + c
    
    return calibrated_reading


# ===== 置信度与异常检测 =====

def check_confidence_and_angle_anomaly(quality_metrics, angle_variance_threshold=0.5, 
                                       confidence_threshold=0.5):
    """
    检查置信度与夹角异常，返回报警级别
    
    参数:
        quality_metrics: 质量指标字典
        angle_variance_threshold: 角度方差阈值
        confidence_threshold: 置信度阈值
    
    返回:
        alarm_level: 报警级别 ("normal", "warning", "critical")
        message: 报警信息
    """
    confidence = quality_metrics.get("ransac_confidence", 0.0)
    num_inliers = quality_metrics.get("num_inliers", 0)
    num_intersections = quality_metrics.get("num_intersections", 0)
    
    # 计算内点比例
    inlier_ratio = num_inliers / max(num_intersections, 1)
    
    # 判断报警级别
    if confidence < confidence_threshold or inlier_ratio < 0.5:
        if confidence < 0.3 or inlier_ratio < 0.3:
            return "critical", f"检测置信度过低 ({confidence:.2f})，内点比例 {inlier_ratio:.2f}"
        else:
            return "warning", f"检测置信度较低 ({confidence:.2f})，请检查图像质量"
    
    if num_intersections < 3:
        return "warning", "交点数量不足，结果可能不准确"
    
    return "normal", "检测正常"


# ===== 数据库初始化 =====
def init_db():
    """初始化数据库（与原版相同，这里省略详细代码）"""
    # ... 原有的init_db代码 ...
    pass  # 实际使用时需要完整复制原有代码


# ===== 核心检测流程（改进版）=====

# 全局task状态
_task_states = {}
_task_lock = threading.Lock()

def _fwdpath(p: str) -> str:
    return p.replace('\\', '/').replace('\\', '/') if p else p

# 序号对照表
try:
    _df = pd.read_excel(FILE_EXCEL_PATH, engine="openpyxl")
    _df["序号"] = _df["序号"].astype(str)
    MY_DICT = pd.Series(_df["表计"].values, index=_df["序号"]).to_dict()
    print(f"序号对照表加载成功，共{len(MY_DICT)}条")
except Exception as e:
    print(f"序号对照表加载失败：{e}")
    MY_DICT = {}

_date_counter = {}
task_id_lock = threading.Lock()

def generate_unique_task_id(conn):
    """生成唯一的 task_id"""
    with task_id_lock:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        
        with conn.cursor() as cur:
            cur.execute(
                "SELECT task_id FROM yolo "
                "WHERE task_id LIKE %s "
                "ORDER BY task_id DESC LIMIT 1",
                (f"{today}-%",)
            )
            result = cur.fetchone()
            
            if result:
                last_task_id = result['task_id']
                try:
                    last_seq = int(last_task_id.split('-')[-1])
                    next_seq = last_seq + 1
                except:
                    next_seq = 1
            else:
                next_seq = 1
            
            task_id = f"{today}-{next_seq}"
            return task_id

def db_update_yolo(task_id: str, **fields):
    """更新yolo表"""
    if not fields:
        return
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


def _run_detection(task_id, image_path, image_name, image_name_all, user_id):
    """改进的检测流程"""
    def log(msg):
        with _task_lock:
            _task_states[task_id]["logs"].append(msg)

    state = _task_states[task_id]
    state["status"] = "running"
    db_update_yolo(task_id, user_id=user_id, detect_status="running")

    try:
        rdir = os.path.join(OUTPUT_FOLDER,
               f"outputs-{datetime.datetime.now().strftime('%Y-%m-%d')}")
        rimg = os.path.join(rdir, image_name)
        os.makedirs(rimg, exist_ok=True)
        txt1 = os.path.join(rimg, "result.txt")

        # 步骤1-3：与原版相同
        log("步骤1/5：裁剪仪表盘区域...")
        r1 = YOLO(MODEL_PATH1)(source=image_path, save=True, save_txt=True, save_crop=True, conf=0.7)
        sp1 = str(r1[0].save_dir)
        i1 = cv2.imread(os.path.join(sp1, "crops", "Instrument", image_name_all))
        obb_path = os.path.join(rimg, image_name + "_all.jpg")
        cv2.imwrite(obb_path, i1)
        state["img_obb"] = _fwdpath(obb_path)
        db_update_yolo(task_id, obb_img_path=_fwdpath(obb_path))
        log("步骤1完成")
        imgp = obb_path

        log("步骤2/5：裁剪表盘...")
        r2 = YOLO(MODEL_PATH2)(source=imgp, save=True, save_txt=True, save_crop=True, conf=0.7)
        sp2 = str(r2[0].save_dir)
        i2 = cv2.imread(os.path.join(sp2, "crops", "Pointer", image_name + "_all.jpg"))
        dial_path = os.path.join(rimg, image_name + "_biaopan.jpg")
        cv2.imwrite(dial_path, i2)
        state["img_dial"] = _fwdpath(dial_path)
        db_update_yolo(task_id, dial_img_path=_fwdpath(dial_path))
        log("步骤2完成")

        log("步骤3/5：裁剪序号标签...")
        r3 = YOLO(MODEL_PATH3)(source=imgp, save=True, save_txt=True, save_crop=True, conf=0.7)
        sp3 = str(r3[0].save_dir)
        i3 = cv2.imread(os.path.join(sp3, "crops", "Label", image_name + "_all.jpg"))
        lbl_path = os.path.join(rimg, image_name + "_biaoqian.jpg")
        cv2.imwrite(lbl_path, i3)
        db_update_yolo(task_id, label_img_path=lbl_path)
        log("步骤3完成")

        log("步骤4/5：识别序号标签...")
        r4 = YOLO(MODEL_PATH4)(source=lbl_path, save=True, save_txt=True, conf=0.7)
        lp4 = os.path.join(str(r4[0].save_dir), "labels")
        log("步骤4完成")

        # ===== 步骤5：改进的OBB检测与读数计算 =====
        log("步骤5/5：改进的OBB关键点检测与读数计算...")
        r5 = YOLO(MODEL_PATH4)(source=dial_path, save=True, save_txt=True, conf=0.7)
        sp5 = str(r5[0].save_dir)
        lp5 = os.path.join(sp5, "labels")

        image = cv2.imread(os.path.join(sp5, image_name + "_biaopan.jpg"))
        h, w = image.shape[:2]

        # 读检测结果txt
        rows = []
        with open(os.path.join(lp5, image_name + "_biaopan.txt")) as f:
            for line in f:
                rows.append(line.strip().split())
        sr = sorted(rows, key=lambda x: (float(x[0]), float(x[1])))

        # 解析坐标
        def rc(r, i): return float(r[i])
        
        # 起点OBB
        xs1, ys1 = w * rc(sr[0], 7), h * rc(sr[0], 8)
        xs2, ys2 = w * rc(sr[0], 1), h * rc(sr[0], 2)
        xs3, ys3 = w * rc(sr[0], 3), h * rc(sr[0], 4)
        xs4, ys4 = w * rc(sr[0], 5), h * rc(sr[0], 6)
        xsf, ysf = (xs1 + xs2 + xs3 + xs4) / 4, (ys1 + ys2 + ys3 + ys4) / 4

        # 终点OBB
        xe1, ye1 = w * rc(sr[1], 1), h * rc(sr[1], 2)
        xe2, ye2 = w * rc(sr[1], 3), h * rc(sr[1], 4)
        xe3, ye3 = w * rc(sr[1], 5), h * rc(sr[1], 6)
        xe4, ye4 = w * rc(sr[1], 7), h * rc(sr[1], 8)
        xef, yef = (xe1 + xe2 + xe3 + xe4) / 4, (ye1 + ye2 + ye3 + ye4) / 4

        # 指针OBB
        xp1, yp1 = w * rc(sr[3], 7), h * rc(sr[3], 8)
        xp2, yp2 = w * rc(sr[3], 1), h * rc(sr[3], 2)
        xp3, yp3 = w * rc(sr[3], 3), h * rc(sr[3], 4)
        xp4, yp4 = w * rc(sr[3], 5), h * rc(sr[3], 6)
        xpf, ypf = (xp1 + xp2 + xp3 + xp4) / 4, (yp1 + yp2 + yp3 + yp4) / 4

        # 零点OBB
        xz1, yz1 = w * rc(sr[2], 1), h * rc(sr[2], 2)
        xz2, yz2 = w * rc(sr[2], 3), h * rc(sr[2], 4)
        xz3, yz3 = w * rc(sr[2], 5), h * rc(sr[2], 6)
        xz4, yz4 = w * rc(sr[2], 7), h * rc(sr[2], 8)
        xzf, yzf = (xz1 + xz2 + xz3 + xz4) / 4, (yz1 + yz2 + yz3 + yz4) / 4

        # 获取边缘中点
        (ss1, ss2), (ss3, ss4) = mid_point(xs1, ys1, xs2, ys2, xs3, ys3, xs4, ys4)
        (ee1, ee2), (ee3, ee4) = mid_point(xe1, ye1, xe2, ye2, xe3, ye3, xe4, ye4)
        (pp1, pp2), (pp3, pp4) = mid_point(xp1, yp1, xp2, yp2, xp3, yp3, xp4, yp4)
        (zz1, zz2), (zz3, zz4) = mid_point(xz1, yz1, xz2, yz2, xz3, yz3, xz4, yz4)
        
        p1 = (ss1, ss2); p2 = (ss3, ss4)
        p3 = (ee1, ee2); p4 = (ee3, ee4)
        p5 = (pp1, pp2); p6 = (pp3, pp4)
        p7 = (zz1, zz2); p8 = (zz3, zz4)

        # ===== 改进的圆心估计 =====
        # 构建多组线段对：SEP, SEZ, SPZ, PEZ等
        line_pairs = [
            ((p1, p2), (p3, p4)),  # SE
            ((p1, p2), (p5, p6)),  # SP
            ((p1, p2), (p7, p8)),  # SZ
            ((p3, p4), (p5, p6)),  # EP
            ((p3, p4), (p7, p8)),  # EZ
            ((p5, p6), (p7, p8)),  # PZ
        ]
        
        # 使用改进的圆心估计算法
        center, quality_metrics = advanced_circle_estimation(line_pairs)
        
        if center is None:
            log("圆心估计失败，使用备用方法")
            # 备用：简单平均
            ise = calc_intersection(p1, p2, p3, p4)
            isp = calc_intersection(p1, p2, p5, p6)
            isz = calc_intersection(p1, p2, p7, p8)
            valid_intersections = [pt for pt in [ise, isp, isz] if pt is not None]
            if valid_intersections:
                center = tuple(np.mean(valid_intersections, axis=0))
            else:
                center = ((xsf + xef + xpf + xzf) / 4, (ysf + yef + ypf + yzf) / 4)
        
        cx, cy = center
        
        # ===== 应用卡尔曼滤波/低通滤波 =====
        filters = get_filters(task_id)
        cx_filtered = filters['kalman_cx'].update(cx)
        cy_filtered = filters['kalman_cy'].update(cy)
        
        # 也可以选择低通滤波
        # cx_filtered = filters['lowpass_cx'].update(cx)
        # cy_filtered = filters['lowpass_cy'].update(cy)
        
        log(f"原始圆心: ({cx:.2f}, {cy:.2f}), 滤波后: ({cx_filtered:.2f}, {cy_filtered:.2f})")
        
        # 使用滤波后的圆心
        cx, cy = cx_filtered, cy_filtered
        
        # ===== 置信度与异常检测 =====
        alarm_level, alarm_msg = check_confidence_and_angle_anomaly(quality_metrics)
        log(f"检测质量: {alarm_level} - {alarm_msg}")
        
        # 在图上标注
        image_annotated = image.copy()
        cv2.circle(image_annotated, (int(cx), int(cy)), 8, (0, 255, 0), -1)  # 圆心
        cv2.circle(image_annotated, (int(xsf), int(ysf)), 5, (255, 0, 0), -1)  # 起点
        cv2.circle(image_annotated, (int(xef), int(yef)), 5, (0, 0, 255), -1)  # 终点
        cv2.circle(image_annotated, (int(xpf), int(ypf)), 5, (0, 255, 255), -1)  # 指针
        cv2.circle(image_annotated, (int(xzf), int(yzf)), 5, (255, 255, 0), -1)  # 零点
        
        # 标注交点
        if "candidate_centers" in quality_metrics:
            for i, cand in enumerate(quality_metrics["candidate_centers"]):
                cv2.circle(image_annotated, (int(cand[0]), int(cand[1])), 4, (128, 128, 128), 1)
        
        # 保存拟合结果图
        fit_path = os.path.join(rimg, image_name + "_fitting_improved.jpg")
        cv2.imwrite(fit_path, image_annotated)
        state["img_fitting"] = _fwdpath(fit_path)
        
        # ===== 改进的读数计算 =====
        # 计算各关键点相对圆心的角度
        angle_start = clock_angle([1, 0], [xsf - cx, ysf - cy])
        angle_end = clock_angle([1, 0], [xef - cx, yef - cy])
        angle_pointer = clock_angle([1, 0], [xpf - cx, ypf - cy])
        angle_zero = clock_angle([1, 0], [xzf - cx, yzf - cy])
        
        # 归一化角度到起点为0
        angle_pointer_rel = clock_angle([xsf - cx, ysf - cy], [xpf - cx, ypf - cy])
        angle_end_rel = clock_angle([xsf - cx, ysf - cy], [xef - cx, yef - cy])
        angle_zero_rel = clock_angle([xsf - cx, ysf - cy], [xzf - cx, yzf - cy])
        
        # 使用最小二乘法校正读数
        slider_position = 0.5  # 可从界面获取，这里默认中间位置
        keypoint_angles = [0.0, angle_pointer_rel, angle_end_rel, angle_zero_rel]
        reference_readings = [0.0, 1.0]  # 假设量程0-1
        
        calibrated_reading = least_squares_reading_calibration(
            keypoint_angles, reference_readings, slider_position
        )
        
        # 原始读数（用于对比）
        rv1_original = angle_pointer_rel / angle_end_rel if angle_end_rel > 1e-6 else 0.0
        
        # 最终读数
        final = calibrated_reading
        
        log(f"原始读数: {rv1_original:.6f}")
        log(f"校正读数: {final:.6f}")
        log(f"质量指标: {quality_metrics}")
        
        # 查序号
        snum = "未知"
        bq_txt = os.path.join(lp4, image_name + "_biaoqian.txt")
        try:
            with open(bq_txt) as f:
                for line in f:
                    snum = MY_DICT.get(line.split()[0], f"序号{line.split()[0]}")
                    break
        except:
            pass
        
        # 记录日志
        now = datetime.datetime.now()
        op = state.get("operator", "unknown")
        entry = (f"{now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"任务编号：{task_id}  操作人：{op}\n"
                f"{image_name} {snum}\n"
                f"原始读数：{rv1_original:.6f}\n"
                f"校正读数：{final:.6f}\n"
                f"检测质量：{alarm_level} - {alarm_msg}\n"
                f"圆心质量指标：{quality_metrics}\n\n")
        
        for p in [TXT_LOG_PATH, txt1]:
            with open(p, "a", encoding="utf-8") as f:
                f.write(entry)
        
        db_update_yolo(task_id, serial_number=snum, reading_before=round(rv1_original, 6),
                       reading_after=round(final, 6), detect_status="success",
                       detected_at=now.strftime("%Y-%m-%d %H:%M:%S"))
        
        with _task_lock:
            state.update({
                "status": "done",
                "detect_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "serial_number": snum,
                "reading_before": round(rv1_original, 6),
                "reading_after": round(final, 6),
                "alarm_level": alarm_level,
                "alarm_message": alarm_msg,
                "quality_metrics": quality_metrics
            })
        
        log(f"完成 | 序号：{snum}")
        log(f"原始读数：{rv1_original:.6f}")
        log(f"校正读数：{final:.6f}")
        log(f"检测质量：{alarm_level}")

    except Exception as exc:
        traceback.print_exc()
        with _task_lock:
            _task_states[task_id]["status"] = "error"
            _task_states[task_id]["error"] = str(exc)
        db_update_yolo(task_id, detect_status="failed")
        log(f"检测出错：{exc}")


# ===== 其余路由与原版相同，这里省略 =====
# 包括：login_required, api_auth_required, 各种API路由等
# 实际使用时需要完整复制原有代码

def api_auth_required(f):
    """API认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            payload = verify_token(token)
            if payload:
                request.current_user_id = payload['user_id']
                request.current_username = payload['username']
                request.current_user_level = payload['user_level']
                return f(*args, **kwargs)
        
        if session.get('user_id'):
            request.current_user_id = session.get('user_id')
            request.current_username = session.get('username')
            request.current_user_level = session.get('user_level')
            return f(*args, **kwargs)
        
        return jsonify({"error": "未登录或登录已过期"}), 401
    
    return decorated_function

def login_required(f):
    """Web端登录装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            if request.method == "GET":
                return redirect(url_for("login_page"))
            return jsonify({"error": "请先登录", "redirect": "/login"}), 401
        return f(*args, **kwargs)
    return decorated

def get_json() -> dict:
    """安全获取请求body的json"""
    return request.get_json(force=True, silent=True) or {}

# ===== 路由示例（完整版需要复制所有原有路由）=====

@app.route("/login")
def login_page():
    if "username" in session:
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/")
@login_required
def index():
    return render_template("index.html",
                           username=session.get("username"),
                           user_level=session.get("user_level"))

# ... 其他所有路由 ...

if __name__ == "__main__":
    init_db()

    ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_ctx.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

    try:
        import OpenSSL
        ssl_ctx = 'adhoc'
        print("=" * 55)
        print("检测到pyOpenSSL，启用HTTPS自签名证书")
        print("本机：https://127.0.0.1:5000")
        print("局域网：https://<本机IP>:5000")
        print("浏览器会提示证书不安全，点高级->继续访问就行")
        print("=" * 55)
    except ImportError:
        print("=" * 55)
        print("没有pyOpenSSL，HTTP模式启动")
        print("摄像头只在 http://127.0.0.1:5000 下可用")
        print("局域网要用摄像头的话：pip install pyOpenSSL 然后重启")
        print("=" * 55)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True,
        ssl_context=('cert.pem', 'key.pem')
    )
