# APK 下载功能集成指南

## 📋 功能说明

实现扫码下载本地 APK 文件的功能：
- 浏览器扫描二维码 → 下载 APK
- 微信扫描二维码 → 跳转小程序（可选）
- APK 文件存储在项目本地：`./static/downloads/app.apk`

---

## 🚀 快速开始（3步部署）

### 步骤 1：运行部署脚本

```bash
python deploy_apk_download.py
```

这会自动：
- ✅ 创建文件夹 `./static/downloads/`
- ✅ 查找并复制 APK 文件
- ✅ 验证文件完整性

### 步骤 2：替换修改后的文件

将以下文件替换到项目中：

| 文件 | 说明 |
|------|------|
| `qrcode_helper.py` | 替换为 `qrcode_helper_modified.py` |
| `app.py` | 按照 `app_py_modifications.txt` 修改 |

### 步骤 3：启动服务器并测试

```bash
python app.py
```

测试链接：
- 下载 APK: http://localhost:5000/downloads/app.apk
- 生成二维码: http://localhost:5000/api/qrcode

---

## 📝 详细修改说明

### 1. 修改 qrcode_helper.py

**完全替换**原文件为 `qrcode_helper_modified.py`

**关键修改：**
```python
# 原代码（第 18-44 行）
def generate_qrcode_url(miniprogram_path, miniprogram_appid, apk_url):
    # ...
    url = apk_url  # ← 使用外部 URL
    return url

# 新代码
def generate_qrcode_url(miniprogram_path, miniprogram_appid, use_local_apk=True):
    # ...
    if use_local_apk:
        url = url_for('download_apk', _external=True)  # ← 使用本地路由
    return url
```

### 2. 修改 app.py

**在 app.py 中添加以下路由（约第 897 行之前）：**

```python
# ========== APK 下载路由 ==========
@app.route("/downloads/app.apk")
def download_apk():
    """
    APK 文件下载路由
    提供本地 APK 文件下载
    """
    apk_dir = os.path.join(app.root_path, 'static', 'downloads')
    apk_filename = 'app.apk'
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
        as_attachment=True,
        download_name='app.apk',
        mimetype='application/vnd.android.package-archive'
    )
```

**修改原有的二维码路由（约第 897-908 行）：**

```python
@app.route("/api/qrcode")
def api_qrcode():
    """
    生成动态二维码
    微信扫描：跳转到微信小程序
    浏览器扫描：下载本地APK
    """
    return serve_qrcode(
        WECHAT_MINIPROGRAM_PATH, 
        WECHAT_MINIPROGRAM_APPID, 
        use_local_apk=True  # ← 添加这个参数
    )
```

---

## 📁 文件结构

部署完成后的文件结构：

```
项目根目录/
├── app.py                          # Flask 主程序（已修改）
├── qrcode_helper.py                # 二维码辅助模块（已修改）
├── config.py                       # 配置文件
├── deploy_apk_download.py          # 部署脚本
├── static/                         # 静态文件目录
│   └── downloads/                  # APK 下载目录
│       └── app.apk                 # ✅ APK 文件（需要放置）
└── templates/                      # HTML 模板
```

---

## 🔧 手动复制 APK 文件

如果部署脚本未自动找到 APK，需要手动复制：

**源文件位置：**
```
D:\BIYESHEJI\BISHE-004\flask-android-app\android\app\build\outputs\apk\debug\app-debug.apk
```

**目标位置：**
```
./static/downloads/app.apk
```

**PowerShell 命令：**
```powershell
# 创建文件夹
New-Item -ItemType Directory -Force -Path .\static\downloads

# 复制 APK
Copy-Item "D:\BIYESHEJI\BISHE-004\flask-android-app\android\app\build\outputs\apk\debug\app-debug.apk" `
          -Destination ".\static\downloads\app.apk"
```

**Linux/Mac 命令：**
```bash
# 创建文件夹
mkdir -p ./static/downloads

# 复制 APK
cp /path/to/app-debug.apk ./static/downloads/app.apk
```

---

## 🧪 测试步骤

### 1. 测试直接下载

```bash
# 启动服务器
python app.py
```

浏览器访问：
```
http://localhost:5000/downloads/app.apk
```

**预期结果：** 浏览器开始下载 APK 文件

### 2. 测试二维码生成

浏览器访问：
```
http://localhost:5000/api/qrcode
```

**预期结果：** 显示二维码图片

### 3. 测试扫码下载

**方式 1：使用本机（推荐）**
1. 启动服务器：`python app.py`
2. 访问：`http://localhost:5000/api/qrcode`
3. 用手机扫描屏幕上的二维码
4. 如果提示"无法访问"，说明需要使用局域网 IP

**方式 2：使用局域网 IP**
1. 查看本机 IP：
   ```powershell
   ipconfig  # Windows
   ifconfig  # Linux/Mac
   ```
   假设是：`192.168.1.100`

2. 手机访问：
   ```
   https://192.168.1.100:5000/api/qrcode
   ```
   
3. 点击"高级" → "继续访问"（HTTPS 证书警告）

4. 扫描二维码下载 APK

---

## ✅ 验证清单

完成部署后，检查以下项目：

- [ ] 文件夹 `./static/downloads/` 已创建
- [ ] APK 文件 `./static/downloads/app.apk` 存在且大小正常（> 1MB）
- [ ] `qrcode_helper.py` 已替换为修改版
- [ ] `app.py` 已添加 `download_apk()` 路由
- [ ] `app.py` 的 `api_qrcode()` 路由已添加 `use_local_apk=True` 参数
- [ ] 访问 `http://localhost:5000/downloads/app.apk` 可以下载
- [ ] 访问 `http://localhost:5000/api/qrcode` 可以显示二维码
- [ ] 手机扫描二维码可以下载 APK

---

## 🐛 常见问题

### Q1: 访问 /downloads/app.apk 返回 404

**原因：** APK 文件不存在

**解决：**
```bash
# 检查文件是否存在
ls -l ./static/downloads/app.apk  # Linux/Mac
dir .\static\downloads\app.apk    # Windows

# 手动复制 APK 文件
```

### Q2: 扫描二维码无法下载

**原因：** 手机无法访问电脑的 localhost

**解决：** 使用局域网 IP
1. 查看本机 IP：`ipconfig`
2. 手机访问：`https://<本机IP>:5000/api/qrcode`

### Q3: 证书不安全警告

**原因：** 使用自签名 HTTPS 证书

**解决：** 点击"高级" → "继续访问"（仅开发环境）

### Q4: 下载的文件无法安装

**原因：** APK 文件损坏或不完整

**解决：**
1. 检查文件大小是否正常（> 1MB）
2. 重新编译 APK：`cd android && gradlew assembleDebug`
3. 重新复制到 `./static/downloads/app.apk`

### Q5: 二维码生成失败

**原因：** 缺少 qrcode 库

**解决：**
```bash
pip install qrcode pillow --break-system-packages
```

---

## 📊 API 接口说明

### 1. 下载 APK

**接口：** `GET /downloads/app.apk`

**功能：** 下载本地 APK 文件

**返回：** APK 文件流（application/vnd.android.package-archive）

**错误：** 404 - APK 文件不存在

### 2. 生成二维码

**接口：** `GET /api/qrcode`

**功能：** 生成动态二维码（根据 User-Agent 区分微信和浏览器）

**返回：** PNG 图片

**逻辑：**
- 微信浏览器 → 生成小程序跳转链接
- 普通浏览器 → 生成 APK 下载链接

### 3. 获取二维码信息

**接口：** `GET /api/qrcode/info`

**功能：** 获取二维码类型和提示信息（供前端显示）

**返回：** JSON
```json
{
  "type": "browser",
  "title": "扫码下载",
  "description": "下载移动应用APK"
}
```

---

## 🔒 安全建议

### 开发环境
- ✅ 使用 HTTPS 自签名证书（Flask adhoc 模式）
- ✅ APK 存储在本地静态文件夹
- ✅ 仅局域网访问

### 生产环境
- ⚠️ 使用正式 HTTPS 证书
- ⚠️ 配置防火墙限制访问
- ⚠️ 考虑使用 CDN 托管 APK
- ⚠️ 添加下载日志记录
- ⚠️ 限制下载速率

---

## 📞 技术支持

如果遇到问题，请检查：
1. Python 版本 >= 3.7
2. Flask 版本 >= 2.0
3. qrcode 和 Pillow 已安装
4. APK 文件完整且可安装
5. 手机和电脑在同一局域网

---

**部署完成后，享受扫码下载的便利吧！** 🎉
