# Flask后端适配微信小程序指南

本文档说明如何修改现有Flask应用以支持微信小程序。

## 核心改动

### 1. Token认证替代Session

微信小程序不支持Cookie，需要使用Token认证。

#### 安装依赖
```bash
pip install pyjwt
```

#### 修改 app.py

在文件顶部添加：

```python
import jwt
from datetime import datetime, timedelta

# JWT配置
JWT_SECRET_KEY = "your_secret_key_change_in_production"
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 7 * 24 * 60 * 60  # 7天

def create_token(user_id, username, user_level):
    """生成JWT Token"""
    payload = {
        'user_id': user_id,
        'username': username,
        'user_level': user_level,
        'exp': datetime.utcnow() + timedelta(seconds=JWT_EXP_DELTA_SECONDS),
        'iat': datetime.utcnow()
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

def verify_token(token):
    """验证并解析Token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token过期
    except jwt.InvalidTokenError:
        return None  # Token无效

def get_token_from_request():
    """从请求头获取Token"""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]  # 移除 'Bearer ' 前缀
    return None
```

#### 修改登录装饰器

```python
# 原有的 login_required 装饰器改为支持Token
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 优先使用Token认证（小程序）
        token = get_token_from_request()
        if token:
            payload = verify_token(token)
            if payload:
                # 将用户信息存储到 g 对象中
                from flask import g
                g.user_id = payload['user_id']
                g.username = payload['username']
                g.user_level = payload['user_level']
                return f(*args, **kwargs)
            else:
                return jsonify({"error": "Token无效或已过期"}), 401
        
        # 回退到Session认证（Web端）
        if 'user_id' in session:
            from flask import g
            g.user_id = session.get('user_id')
            g.username = session.get('username')
            g.user_level = session.get('user_level')
            return f(*args, **kwargs)
        
        return jsonify({"error": "未授权"}), 401
    return decorated_function
```

#### 修改登录接口

```python
@app.route("/login", methods=["POST"])
def login():
    # 兼容Web表单和小程序JSON
    if request.is_json:
        data = request.get_json()
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
    else:
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
    
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, password, user_level FROM `user` WHERE username=%s",
                    (username,)
                )
                user = cur.fetchone()
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": f"数据库错误: {str(e)}"}), 500
    
    if not user or not _check_pw(password, user["password"]):
        return jsonify({"error": "用户名或密码错误"}), 401
    
    # 小程序返回Token
    if request.is_json:
        token = create_token(user["id"], user["username"], user["user_level"])
        return jsonify({
            "message": "登录成功",
            "token": token,
            "user_id": user["id"],
            "username": user["username"],
            "user_level": user["user_level"]
        })
    
    # Web返回Session
    else:
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["user_level"] = user["user_level"]
        return redirect(url_for("index_page"))
```

#### 修改注册接口

```python
@app.route("/register", methods=["POST"])
def register():
    # 兼容Web表单和小程序JSON
    if request.is_json:
        data = request.get_json()
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
    else:
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
    
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    
    if len(password) < 6:
        return jsonify({"error": "密码长度至少6位"}), 400
    
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                # 检查用户名是否存在
                cur.execute("SELECT id FROM `user` WHERE username=%s", (username,))
                if cur.fetchone():
                    return jsonify({"error": "用户名已存在"}), 400
                
                # 创建用户
                hashed_pw = _hash_pw(password)
                cur.execute(
                    "INSERT INTO `user`(username, password, user_level) VALUES(%s, %s, %s)",
                    (username, hashed_pw, "user")
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": f"注册失败: {str(e)}"}), 500
    
    return jsonify({"message": "注册成功"})
```

#### 修改退出接口

```python
@app.route("/logout", methods=["POST", "GET"])
def logout():
    # 清除Session（Web端）
    session.clear()
    
    # 小程序端直接返回成功（Token在客户端管理）
    if request.method == "POST" or request.is_json:
        return jsonify({"message": "退出成功"})
    
    # Web端重定向
    return redirect(url_for("login_page"))
```

### 2. 修改数据库操作

在所有使用 `session.get()` 的地方改为使用 `g` 对象：

```python
# 原代码
user_id = session.get("user_id")
username = session.get("username")

# 修改为
from flask import g
user_id = getattr(g, 'user_id', None)
username = getattr(g, 'username', None)
```

示例修改位置：
- `/api/detect` 接口
- `/confirm` 接口  
- `/modify` 接口
- `/api/history` 接口

### 3. 检测接口适配

```python
@app.route("/api/detect", methods=["POST"])
@login_required
def api_detect():
    from flask import g
    user_id = g.user_id
    
    # 获取上传的文件
    if 'file' not in request.files:
        return jsonify({"error": "未上传图片"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    
    # 获取序列号（可选）
    serial_number = request.form.get('serial_number', '').strip()
    
    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # 生成任务ID
    task_id = f"task_{timestamp}_{user_id}"
    
    # 插入数据库
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO `yolo`(task_id, user_id, serial_number, original_img_path, detect_status) "
                    "VALUES(%s, %s, %s, %s, %s)",
                    (task_id, user_id, serial_number or None, filepath, 'pending')
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": f"数据库错误: {str(e)}"}), 500
    
    # 启动后台检测线程
    threading.Thread(target=do_yolo_detect, args=(task_id, filepath, serial_number)).start()
    
    return jsonify({
        "task_id": task_id,
        "message": "检测任务已创建"
    })
```

### 4. 结果查询接口

```python
@app.route("/api/result", methods=["GET"])
@login_required
def api_result():
    task_id = request.args.get("task_id", "").strip()
    if not task_id:
        return jsonify({"error": "task_id参数不能为空"}), 400
    
    try:
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT task_id, serial_number, original_img_path, dial_img_path, "
                    "label_img_path, obb_img_path, reading_before, reading_after, "
                    "detect_status, is_confirmed, confirmed_at, detected_at, created_at "
                    "FROM `yolo` WHERE task_id=%s",
                    (task_id,)
                )
                result = cur.fetchone()
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": f"数据库错误: {str(e)}"}), 500
    
    if not result:
        return jsonify({"error": "任务不存在"}), 404
    
    # 转换datetime为字符串
    for key in ['confirmed_at', 'detected_at', 'created_at']:
        if result.get(key):
            result[key] = str(result[key])
    
    # 转换Decimal为float
    for key in ['reading_before', 'reading_after']:
        if result.get(key) is not None:
            result[key] = float(result[key])
    
    return jsonify(result)
```

### 5. CORS配置（如果需要）

如果小程序和后端不在同一域名，需要配置CORS：

```bash
pip install flask-cors
```

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
```

### 6. HTTPS配置

生产环境必须使用HTTPS：

#### 生成自签名证书（开发测试）
```bash
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

#### 启动HTTPS服务
```python
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        ssl_context=('cert.pem', 'key.pem')
    )
```

#### 使用Nginx反向代理（推荐）
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 完整的修改后app.py示例

```python
# 在原有app.py基础上添加以下内容

import jwt
from datetime import datetime, timedelta
from flask import g

# JWT配置
JWT_SECRET_KEY = "your_secret_key_change_in_production"
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 7 * 24 * 60 * 60

def create_token(user_id, username, user_level):
    payload = {
        'user_id': user_id,
        'username': username,
        'user_level': user_level,
        'exp': datetime.utcnow() + timedelta(seconds=JWT_EXP_DELTA_SECONDS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_token(token):
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except:
        return None

def get_token_from_request():
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]
    return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = get_token_from_request()
        if token:
            payload = verify_token(token)
            if payload:
                g.user_id = payload['user_id']
                g.username = payload['username']
                g.user_level = payload['user_level']
                return f(*args, **kwargs)
            else:
                return jsonify({"error": "Token无效或已过期"}), 401
        
        if 'user_id' in session:
            g.user_id = session.get('user_id')
            g.username = session.get('username')
            g.user_level = session.get('user_level')
            return f(*args, **kwargs)
        
        return jsonify({"error": "未授权"}), 401
    return decorated_function

# 其他接口按照上述方式修改...
```

## 测试步骤

1. **安装依赖**
```bash
pip install pyjwt
```

2. **修改代码**
按照上述指南修改app.py

3. **重启服务**
```bash
python app.py
```

4. **测试Token生成**
```python
# 在Python终端测试
from app import create_token, verify_token
token = create_token(1, "admin", "super_admin")
print(token)
print(verify_token(token))
```

5. **测试小程序登录**
在小程序中登录，检查是否返回token

6. **测试API调用**
检查带token的请求是否正常工作

## 注意事项

1. ⚠️ **JWT_SECRET_KEY** 必须修改为强密码
2. ⚠️ 生产环境必须使用 **HTTPS**
3. ⚠️ Token过期时间根据需求调整
4. ⚠️ 敏感操作建议缩短token有效期
5. ⚠️ 记得处理token刷新逻辑

## 故障排查

### Token验证失败
- 检查JWT_SECRET_KEY是否一致
- 检查token格式是否正确（Bearer token）
- 检查token是否过期

### 数据库连接失败
- 确认MySQL服务运行
- 检查数据库配置
- 查看防火墙设置

### 图片无法访问
- 检查图片路径
- 确认/image接口工作正常
- 验证HTTPS证书

## 完成检查清单

- [ ] 安装pyjwt
- [ ] 添加Token生成和验证函数
- [ ] 修改login_required装饰器
- [ ] 修改登录接口返回Token
- [ ] 修改所有接口的session为g对象
- [ ] 配置HTTPS
- [ ] 测试小程序登录
- [ ] 测试检测功能
- [ ] 测试历史记录
- [ ] 部署到生产环境
