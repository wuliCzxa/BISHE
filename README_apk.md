# Flask 套壳打包 APK 说明文档

> 使用 Python 脚本将 Flask Web 应用自动打包为 Android APK（基于 Capacitor）

---

## 目录

- [项目简介](#项目简介)
- [原理说明](#原理说明)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [构建流程详解](#构建流程详解)
- [常见问题](#常见问题)
- [注意事项](#注意事项)

---

## 项目简介

`build.py` 是一个全自动化脚本，可将已部署的 Flask Web 服务一键打包为 Android APK 安装包。最终 APK 启动后会在原生 WebView 中加载指定的 Flask 服务器地址，实现"套壳"效果。

```
Flask 服务器（远程/本地） ──► Capacitor WebView ──► Android APK
```

---

## 原理说明

脚本使用 [Capacitor](https://capacitorjs.com/) 框架将 Web 页面包装为原生 Android 应用：

1. 创建一个最简 HTML 页面（`www/index.html`），内嵌 `<iframe>` 指向 Flask 服务器
2. 通过 Capacitor 将该 HTML 包装为 Android 项目
3. 使用 Android Studio 或 Gradle 编译生成 APK

---

## 环境要求

| 依赖项 | 版本要求 | 说明 |
|--------|----------|------|
| Python | 3.6+ | 运行构建脚本 |
| Node.js | 14+ | Capacitor CLI 依赖 |
| npm | 6+ | 包管理器 |
| Android Studio | 最新稳定版 | 推荐构建方式 |
| JDK | 11 或 17 | Android 编译需要 |

**环境变量要求：**

- `ANDROID_HOME` 须指向 Android SDK 目录（例如 `C:\Users\<用户名>\AppData\Local\Android\Sdk`）
- Android SDK 的 `platform-tools` 目录须加入系统 `PATH`

> 脚本启动时会自动检测上述依赖，缺失时给出提示。

---

## 快速开始

### 1. 修改配置

打开 `build.py`，编辑顶部配置区：

```python
APP_NAME   = "仪态万象"                                    # 应用显示名称
APP_ID     = "com.mycompany.flaskapp"                      # Android 包名（需唯一）
SERVER_URL = "https://calamari-scorecard-unsolved.ngrok-free.dev"  # Flask 服务器地址
PROJECT_DIR = "flask-android-app-ytwx"                    # 本地项目目录名
```

### 2. 运行脚本

```bash
python build.py
```

脚本启动后会依次提示确认配置、检测环境，然后自动完成所有构建步骤。

### 3. 选择构建方式

脚本最后提示选择构建方式：

```
1. Android Studio（推荐）—— 脚本自动打开 IDE，手动点击 Build > Build APK
2. 命令行构建 ————————————— 脚本自动执行 gradlew.bat assembleDebug
```

### 4. 获取 APK

构建成功后，APK 位于：

```
flask-android-app-ytwx\android\app\build\outputs\apk\debug\app-debug.apk
```

---

## 配置说明

### capacitor.config.json（自动生成）

| 字段 | 说明 |
|------|------|
| `appId` | Android 包名，格式为反向域名，需全局唯一 |
| `appName` | 应用安装后显示的名称 |
| `webDir` | 本地静态资源目录（固定为 `www`） |
| `server.url` | 远程 Flask 服务器地址，WebView 将直接加载此 URL |
| `android.allowMixedContent` | 允许混合内容（HTTP/HTTPS），调试时可开启 |

### www/index.html（自动生成）

- 使用全屏 `<iframe>` 加载 `SERVER_URL`
- 包含加载提示文字"正在连接服务器..."
- 连接超时（10秒）后显示错误提示

> ⚠️ **已知 Bug**：`index.html` 中的超时回调引用了未定义的 `document.getElementById('error')`，且存在多余的 `~` 字符，实际超时提示不会生效，不影响主要功能，如需修复可手动编辑 `www/index.html`。

---

## 构建流程详解

脚本按以下顺序自动执行 8 个步骤：

```
步骤1  检查 Node.js 与 npm 是否可用
步骤2  创建项目目录（PROJECT_DIR）
步骤3  npm init 初始化项目
步骤4  安装 @capacitor/core、@capacitor/cli、@capacitor/android
步骤5  生成 capacitor.config.json
步骤6  生成 www/index.html（WebView 入口页）
步骤7  执行 npx cap add android（添加 Android 平台）
步骤8  执行 npx cap sync（同步 Web 资源到 Android 项目）
       └── 选择构建方式 → 生成 APK
```

---

## 常见问题

**Q：运行脚本时提示"未找到 Node.js"？**  
A：访问 [https://nodejs.org/](https://nodejs.org/) 下载安装 LTS 版本，安装后重新打开终端运行脚本。

**Q：提示"未检测到 ANDROID_HOME"？**  
A：安装 Android Studio 后，在系统环境变量中添加 `ANDROID_HOME`，值为 SDK 路径（如 `C:\Users\用户名\AppData\Local\Android\Sdk`）。

**Q：命令行构建时 `gradlew.bat` 报错？**  
A：优先使用 Android Studio 构建（选项1），更稳定。命令行方式需要正确配置 JDK 版本。

**Q：APP 安装后打开空白或一直显示"正在连接服务器..."？**  
A：检查 `SERVER_URL` 是否可从手机网络访问。若使用 ngrok，需确认 ngrok 隧道处于运行状态且地址未变更。

**Q：目录已存在，如何重新构建？**  
A：脚本会询问是否删除重建（`y/n`），输入 `y` 清空后重新构建，输入 `n` 则复用现有目录（跳过已完成的步骤）。

**Q：需要发布正式版 APK 怎么办？**  
A：当前脚本生成的是 **debug APK**，仅供测试使用。发布 Google Play 或正式分发需在 Android Studio 中进行签名打包（Build > Generate Signed Bundle/APK）。

---

## 注意事项

- **服务器地址变更**：若 Flask 服务器地址（尤其是 ngrok 免费地址）发生变化，需修改 `SERVER_URL` 后重新执行脚本或手动更新 `capacitor.config.json` 并重新 `sync`。
- **HTTPS 要求**：Android 9+ 默认禁止 HTTP 明文请求。生产环境请确保 `SERVER_URL` 使用 HTTPS。
- **包名唯一性**：`APP_ID` 一旦发布不可更改，正式应用请使用自己的域名反转格式，如 `com.yourcompany.appname`。
- **调试版本限制**：`assembleDebug` 生成的 APK 带有调试签名，部分应用市场不接受，仅限测试分发。
