# 动态二维码功能集成指南

本指南将帮助你为指针式仪表读数识别系统添加动态二维码功能。

## 功能说明

- **微信扫描**：自动识别微信浏览器，跳转到微信小程序
- **浏览器扫描**：非微信环境下，提供APK下载链接
- **动态切换**：根据User-Agent自动切换二维码内容

---

## 安装步骤

### 第一步：安装依赖

```bash
pip install qrcode Pillow --break-system-packages
```

如果使用虚拟环境：
```bash
pip install qrcode Pillow
```

### 第二步：更新配置文件

**方式A：直接修改现有的 config.py**

在 `config.py` 文件末尾添加以下内容：

```python
# ---- 二维码配置 ----
# 微信小程序配置
WECHAT_MINIPROGRAM_PATH = "pages/index/index"  # 小程序页面路径
WECHAT_MINIPROGRAM_APPID = "wxabcd1234efgh5678"  # 替换成你的小程序AppID

# APK下载链接
APK_DOWNLOAD_URL = "https://your-domain.com/downloads/app.apk"  # 替换成实际下载链接
# 或者使用本地路径：APK_DOWNLOAD_URL = "/static/downloads/app.apk"
```

**方式B：使用新的 config.py**

将项目提供的 `config.py` 文件替换现有文件，记得修改：
- 数据库密码 `DB_PASSWORD`
- 小程序AppID `WECHAT_MINIPROGRAM_APPID`
- APK下载链接 `APK_DOWNLOAD_URL`

### 第三步：添加二维码辅助模块

将 `qrcode_helper.py` 文件复制到项目根目录（与 app.py 同级）。

### 第四步：修改 app.py

**在文件顶部的导入部分（约第17行）添加：**

```python
from qrcode_helper import serve_qrcode, get_qrcode_info
```

**在配置读取部分（约第32-61行）添加：**

```python
try:
    import config as _cfg
    # ... 现有配置 ...
    
    # 添加二维码配置读取
    WECHAT_MINIPROGRAM_PATH = getattr(_cfg, 'WECHAT_MINIPROGRAM_PATH', 'pages/index/index')
    WECHAT_MINIPROGRAM_APPID = getattr(_cfg, 'WECHAT_MINIPROGRAM_APPID', 'your_miniprogram_appid')
    APK_DOWNLOAD_URL = getattr(_cfg, 'APK_DOWNLOAD_URL', 'https://your-domain.com/downloads/app.apk')
    
    print("二维码配置加载成功")
except (ImportError, AttributeError) as _ce:
    # ... 现有错误处理 ...
    WECHAT_MINIPROGRAM_PATH = "pages/index/index"
    WECHAT_MINIPROGRAM_APPID = "your_miniprogram_appid"
    APK_DOWNLOAD_URL = "https://your-domain.com/downloads/app.apk"
```

**在路由部分（约第900行，@app.route("/image/...") 之前）添加：**

```python
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
        APK_DOWNLOAD_URL
    )


@app.route("/api/qrcode/info")
def api_qrcode_info():
    """
    获取二维码信息（用于前端显示提示）
    """
    from flask import jsonify
    return jsonify(get_qrcode_info())
```

### 第五步：更新 login.html

将项目提供的 `login.html` 文件替换现有的登录页面。

或者手动集成：在左侧面板（.left-panel）中添加二维码卡片HTML（见文件中的 qrcode-card 部分）。

### 第六步：更新 index.html

**选项A：浮动二维码按钮（推荐）**

1. 在 `<style>` 标签末尾添加 `index_qrcode_integration.txt` 中的CSS样式
2. 在 `</body>` 之前添加HTML代码
3. 在 `<script>` 末尾添加JavaScript代码

**选项B：独立二维码页面**

在主页面添加一个链接按钮跳转到二维码页面。

---

## 配置说明

### 微信小程序配置

#### 方案1：使用小程序URL Scheme（推荐）

1. 在微信公众平台获取小程序URL Scheme
2. 修改 `qrcode_helper.py` 中的 `generate_qrcode_url` 函数：

```python
if is_wechat_browser():
    # 使用你的小程序URL Scheme
    url = f"weixin://dl/business/?t={miniprogram_appid}&page={miniprogram_path}"
```

#### 方案2：使用H5中间页

1. 创建一个H5页面，使用微信JSSDK跳转小程序
2. 将APK_DOWNLOAD_URL设置为该H5页面地址

### APK下载配置

#### 本地文件方式

1. 在项目根目录创建 `static/downloads` 文件夹
2. 将APK文件放入该文件夹
3. 在 app.py 中添加静态文件路由：

```python
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join('static', filename))
```

4. 配置：`APK_DOWNLOAD_URL = "/static/downloads/app.apk"`

#### 云存储方式

使用CDN或云存储服务：
- 七牛云：`https://xxx.qiniucdn.com/app.apk`
- 阿里云OSS：`https://xxx.oss-cn-beijing.aliyuncs.com/app.apk`
- 腾讯云COS：`https://xxx.cos.ap-guangzhou.myqcloud.com/app.apk`

---

## 测试验证

### 1. 启动服务

```bash
python app.py
```

### 2. 访问二维码API

浏览器访问：
- 二维码图片：`http://localhost:5000/api/qrcode`
- 二维码信息：`http://localhost:5000/api/qrcode/info`

### 3. 微信测试

1. 使用微信扫描二维码，应跳转到小程序（需要配置正确的AppID）
2. 如果小程序未发布，需要添加体验者

### 4. 浏览器测试

1. 使用手机浏览器扫描二维码
2. 应该打开APK下载页面

---

## 常见问题

### Q1: 二维码无法显示

**检查项：**
- 确认已安装 qrcode 和 Pillow 库
- 检查 app.py 中是否正确导入 qrcode_helper
- 查看浏览器控制台是否有错误

### Q2: 微信扫码后无法跳转

**原因：**
- 小程序AppID配置错误
- 小程序未发布或未添加体验者
- URL Scheme未正确配置

**解决方案：**
1. 检查 `WECHAT_MINIPROGRAM_APPID` 是否正确
2. 在微信公众平台检查小程序状态
3. 使用微信提供的URL Scheme生成工具

### Q3: APK下载失败

**检查项：**
- APK文件路径是否正确
- 静态文件路由是否配置
- 云存储链接是否有效

### Q4: User-Agent检测不准确

**优化方案：**

修改 `qrcode_helper.py` 中的 `is_wechat_browser` 函数：

```python
def is_wechat_browser():
    user_agent = request.headers.get('User-Agent', '').lower()
    # 微信浏览器特征：micromessenger
    # 企业微信：wxwork
    return 'micromessenger' in user_agent or 'wxwork' in user_agent
```

---

## 高级配置

### 1. 自定义二维码样式

在 `qrcode_helper.py` 中修改 `create_qrcode_image` 函数：

```python
# 添加Logo
from PIL import Image
logo = Image.open('logo.png')
# ... 将logo嵌入二维码中心 ...

# 自定义颜色
img = qr.make_image(fill_color="#2563eb", back_color="#eff6ff")
```

### 2. 统计二维码扫描次数

在 app.py 中添加统计路由：

```python
@app.route("/api/qrcode")
def api_qrcode():
    # 记录扫描
    log_qrcode_scan(request.headers.get('User-Agent'))
    
    return serve_qrcode(
        WECHAT_MINIPROGRAM_PATH,
        WECHAT_MINIPROGRAM_APPID,
        APK_DOWNLOAD_URL
    )
```

### 3. 多端适配

针对不同设备生成不同内容：

```python
def generate_qrcode_url(miniprogram_path, miniprogram_appid, apk_url):
    user_agent = request.headers.get('User-Agent', '').lower()
    
    if 'micromessenger' in user_agent:
        return f"weixin://dl/business/?t={miniprogram_appid}"
    elif 'android' in user_agent:
        return apk_url  # Android APK
    elif 'iphone' in user_agent or 'ipad' in user_agent:
        return "https://apps.apple.com/app/your-app"  # iOS App Store
    else:
        return apk_url  # 默认下载链接
```

---

## 文件清单

完成集成后，你应该有以下文件：

```
项目根目录/
├── app.py                      # 主程序（已修改）
├── config.py                   # 配置文件（已修改）
├── qrcode_helper.py            # 二维码辅助模块（新增）
├── templates/
│   ├── login.html              # 登录页面（已修改）
│   └── index.html              # 主页面（已修改）
└── static/                     # 静态文件目录（可选）
    └── downloads/
        └── app.apk             # APK文件
```

---

## 技术支持

如遇问题，请检查：
1. Python版本 >= 3.7
2. Flask版本 >= 2.0
3. 所有依赖库已安装
4. 配置文件格式正确

祝你集成顺利！🎉
