# 仪态万象微信小程序

基于YOLOv8深度学习模型的指针式仪表智能识别微信小程序，与Flask后端完全对应。

## 功能特性

### 核心功能
- ✅ **用户认证系统** - 登录、注册、权限管理
- ✅ **智能检测** - 拍照/上传图片进行YOLO检测
- ✅ **实时反馈** - 检测进度实时显示，轮询获取结果
- ✅ **结果管理** - 查看、修改、确认检测结果
- ✅ **历史记录** - 分页查询检测历史
- ✅ **趋势分析** - 序列号历史数据可视化图表
- ✅ **操作日志** - 查看和清除操作日志

### 技术特点
- 📱 原生微信小程序开发
- 🎨 精美UI设计，渐变色主题
- 🔄 完整的状态管理
- 📊 Canvas绘制趋势图表
- 🚀 请求封装，统一错误处理
- 🔐 Token认证机制

## 项目结构

```
miniprogram/
├── pages/                  # 页面目录
│   ├── login/             # 登录页面
│   ├── register/          # 注册页面
│   ├── index/             # 首页（检测页面）
│   ├── history/           # 历史记录页面
│   ├── detail/            # 检测详情页面
│   ├── profile/           # 个人中心页面
│   └── chart/             # 趋势图表页面
├── utils/                 # 工具模块
│   ├── request.js         # 网络请求封装
│   ├── api.js             # API接口定义
│   └── util.js            # 通用工具函数
├── images/                # 图片资源
├── app.js                 # 小程序入口
├── app.json               # 小程序配置
└── app.wxss               # 全局样式
```

## 快速开始

### 1. 环境准备

- **微信开发者工具** - 下载安装最新版
- **Flask后端服务** - 确保后端服务已运行
- **HTTPS证书** - 生产环境需要配置HTTPS

### 2. 配置服务器地址

编辑 `app.js` 文件，修改服务器地址：

```javascript
globalData: {
  // 开发环境（需要在开发者工具中勾选"不校验合法域名"）
  serverUrl: 'http://localhost:5000',
  
  // 或生产环境
  // serverUrl: 'https://your-domain.com',
  // serverUrl：'https://192.168.45.79:5000'
}
```

### 3. 后端API适配

确保Flask后端提供以下接口：

#### 用户接口
- `POST /login` - 登录
- `POST /register` - 注册  
- `POST /logout` - 退出

#### 检测接口
- `POST /api/detect` - 上传图片检测
- `GET /api/result?task_id=xxx` - 查询检测结果
- `POST /confirm` - 确认结果
- `POST /modify` - 修改读数

#### 历史接口
- `GET /api/history?page=1&size=20` - 历史记录列表
- `GET /api/serial_history?serial=xxx&limit=60` - 序列号历史

#### 日志接口
- `GET /get_log` - 获取日志
- `POST /clear` - 清除日志

#### 图片服务
- `GET /image/<path>` - 图片访问

### 4. 登录接口返回格式

后端登录接口需要返回以下格式：

```json
{
  "message": "登录成功",
  "token": "your_jwt_token_here",
  "user_id": 1,
  "username": "admin",
  "user_level": "super_admin"
}
```

### 5. 微信开发者工具配置

1. 打开微信开发者工具
2. 导入项目，选择 `miniprogram` 目录
3. 填写 AppID（测试可使用测试号）
4. 开发环境勾选：
   - ☑️ 不校验合法域名、web-view（业务域名）、TLS版本以及HTTPS证书
   - ☑️ 启用代码压缩上传

### 6. 运行项目

1. 启动Flask后端服务
```bash
python app.py
```

2. 在微信开发者工具中点击"编译"
3. 使用默认账号登录测试：
   - 用户名：`admin`
   - 密码：`admin123`

## 核心流程说明

### 检测流程

1. **选择图片** - 拍照或从相册选择
2. **上传检测** - 调用 `/api/detect` 接口
3. **轮询结果** - 每2秒调用 `/api/result` 查询状态
4. **显示结果** - 展示检测图片和读数
5. **确认修正** - 可修改读数并确认

### 认证机制

小程序使用Token认证：
- 登录成功后将token存储在本地
- 每次请求在header中携带token
- token失效时自动跳转登录页

### 数据同步

- 使用 `wx.request` 进行网络请求
- 统一的错误处理和loading提示
- 支持请求拦截和响应拦截

## 生产环境部署

### 1. 域名配置

在微信公众平台配置服务器域名：
- request合法域名：`https://your-domain.com`
- uploadFile合法域名：`https://your-domain.com`
- downloadFile合法域名：`https://your-domain.com`

### 2. HTTPS配置

Flask后端必须启用HTTPS：

```python
# app.py
if __name__ == "__main__":
    app.run(
        host="0.0.0.0", 
        port=5000, 
        ssl_context=('cert.pem', 'key.pem')  # SSL证书
    )
```

### 3. 后端Session适配

微信小程序不支持Cookie，需要修改Flask的session机制为Token：

```python
# 添加到app.py

from functools import wraps
import jwt
import datetime

SECRET_KEY = "your_secret_key"

def create_token(user_id, username, user_level):
    """生成JWT Token"""
    payload = {
        'user_id': user_id,
        'username': username,
        'user_level': user_level,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_token(token):
    """验证Token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except:
        return None

def login_required_token(f):
    """Token验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "未授权"}), 401
        
        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Token无效"}), 401
        
        # 将用户信息注入到request对象
        request.user_id = payload['user_id']
        request.username = payload['username']
        request.user_level = payload['user_level']
        
        return f(*args, **kwargs)
    return decorated_function

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    # 验证用户（原有逻辑）
    # ...
    
    # 生成Token
    token = create_token(user_id, username, user_level)
    
    return jsonify({
        "message": "登录成功",
        "token": token,
        "user_id": user_id,
        "username": username,
        "user_level": user_level
    })
```

### 4. 图片访问适配

确保图片路径可以通过HTTP访问：

```python
@app.route("/image/<path:filepath>")
def serve_image(filepath):
    # 验证路径安全性
    # 返回图片文件
    return send_file(filepath)
```

## 常见问题

### Q1: 开发时提示"不在以下request合法域名列表中"
**A:** 在微信开发者工具中勾选"不校验合法域名"

### Q2: 上传图片失败
**A:** 检查后端是否正确处理 `multipart/form-data` 格式

### Q3: Token认证失败
**A:** 确认后端已实现Token验证，header格式为 `Authorization: Bearer <token>`

### Q4: 图片无法显示
**A:** 检查图片路径是否正确，确保 `/image/<path>` 接口可访问

### Q5: Canvas图表不显示
**A:** 检查数据格式，确保records数组不为空

## API接口详细说明

### 登录接口
```http
POST /login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}

Response:
{
  "message": "登录成功",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "user_id": 1,
  "username": "admin",
  "user_level": "super_admin"
}
```

### 检测接口
```http
POST /api/detect
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: <image_file>
serial_number: "SN001" (optional)

Response:
{
  "task_id": "abc123",
  "message": "检测任务已创建"
}
```

### 查询结果接口
```http
GET /api/result?task_id=abc123
Authorization: Bearer <token>

Response:
{
  "task_id": "abc123",
  "serial_number": "SN001",
  "detect_status": "success",
  "reading_before": 123.456,
  "reading_after": 123.456,
  "is_confirmed": false,
  "original_img_path": "uploads/xxx.jpg",
  "dial_img_path": "outputs/xxx_dial.jpg",
  "label_img_path": "outputs/xxx_label.jpg",
  "obb_img_path": "outputs/xxx_obb.jpg",
  "detected_at": "2025-01-15 10:30:00",
  "created_at": "2025-01-15 10:29:00"
}
```

## 性能优化建议

1. **图片压缩** - 上传前使用 `sizeType: ['compressed']`
2. **请求缓存** - 对不常变化的数据进行缓存
3. **分页加载** - 历史记录使用分页和下拉加载
4. **图片懒加载** - 使用 `lazy-load` 属性
5. **减少setData** - 合并多次setData操作

## 维护说明

### 版本更新
在 `pages/profile/profile.wxml` 中更新版本号

### 日志管理
定期清理操作日志，避免日志文件过大

### 数据备份
定期备份数据库，特别是检测历史数据

## 技术支持

- 微信开发文档：https://developers.weixin.qq.com/miniprogram/dev/
- Flask文档：https://flask.palletsprojects.com/
- YOLO文档：https://docs.ultralytics.com/

## License

© 2026 仪态万象. All rights reserved.
