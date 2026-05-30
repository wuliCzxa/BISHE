# flask-android-app 项目说明文档

> 本项目是在 `build.py` 自动生成的 Capacitor 套壳基础上，手动扩展了多项原生功能，使安卓客户端能正常支持摄像头、麦克风、文件上传、SSL 证书信任等能力。

---

## 目录

- [项目结构](#项目结构)
- [与 build.py 的关系](#与-buildpy-的关系)
- [新增功能说明](#新增功能说明)
- [关键文件详解](#关键文件详解)
- [构建与使用](#构建与使用)
- [修改服务器地址](#修改服务器地址)
- [权限说明](#权限说明)
- [版本信息](#版本信息)
- [注意事项](#注意事项)

---

## 项目结构

```
flask-android-app/
├── capacitor.config.json          # Capacitor 全局配置（服务器地址）
├── www/
│   └── index.html                 # WebView 入口页（iframe 加载 Flask）
├── android/
│   ├── app/
│   │   ├── build.gradle           # 应用级构建配置（版本号、SDK版本、依赖）
│   │   └── src/main/
│   │       ├── AndroidManifest.xml            # 权限声明 & 应用配置
│   │       ├── assets/
│   │       │   ├── capacitor.config.json      # 同步进 Android 的 Capacitor 配置
│   │       │   └── public/index.html          # 打包进 APK 的入口 HTML
│   │       ├── java/com/mycompany/flaskapp/
│   │       │   └── MainActivity.java          # 核心 Native 扩展代码（★ 重点修改）
│   │       └── res/xml/
│   │           └── network_security_config.xml # 网络安全配置（证书信任）
│   ├── variables.gradle           # SDK 版本与依赖版本统一管理
│   └── build.gradle               # 项目级 Gradle 配置
└── node_modules/                  # Capacitor npm 依赖（@capacitor/android 等）
```

---

## 与 build.py 的关系

`build.py` 负责搭建最基础的 Capacitor 套壳框架，生成可以运行的 APK 骨架。本项目在此基础上针对实际应用场景（仪态万象）进行了深度定制：

| 内容 | build.py 生成 | 本项目手动扩展 |
|------|--------------|--------------|
| Capacitor 框架初始化 | ✅ | — |
| www/index.html 基础套壳 | ✅ | 保留并修复 |
| MainActivity（纯转发） | ✅ 极简版 | ✅ 大幅扩展 |
| AndroidManifest 权限 | 基础 | ✅ 补全摄像头/存储/麦克风 |
| SSL 证书信任配置 | ❌ | ✅ 新增 |
| 文件上传支持 | ❌ | ✅ 新增 |
| 硬件加速 | ❌ | ✅ 新增 |
| 动态权限申请（分版本） | ❌ | ✅ 新增 |

---

## 新增功能说明

### 1. SSL 自签名证书信任

Flask 服务器使用 HTTPS 时可能使用自签名证书，Android 默认会拒绝此类连接。本项目在 `MainActivity.java` 中重写了 `onReceivedSslError`，使 WebView 在遇到 SSL 错误时直接放行：

```java
webView.setWebViewClient(new WebViewClient() {
    @Override
    public void onReceivedSslError(WebView view, SslErrorHandler handler, SslError error) {
        handler.proceed(); // 忽略 SSL 证书错误，允许继续加载
    }
});
```

同时在 `network_security_config.xml` 中配置了同时信任系统证书和用户证书（适用于开发环境导入自签名证书的场景）。

> ⚠️ **安全提示**：`handler.proceed()` 会忽略所有 SSL 错误，仅适合内网或受信任环境。生产环境建议使用正规 CA 签发的证书并移除该配置。

---

### 2. 摄像头 & 麦克风权限授权

Flask 页面如需调用摄像头或麦克风（如视频录制、音频采集功能），WebView 默认会拦截浏览器发出的权限请求。本项目通过重写 `onPermissionRequest` 自动授权所有网页发起的硬件权限请求：

```java
webView.setWebChromeClient(new WebChromeClient() {
    @Override
    public void onPermissionRequest(PermissionRequest request) {
        request.grant(request.getResources()); // 自动授权摄像头/麦克风
    }
    // ...
});
```

---

### 3. 文件上传支持

默认 Capacitor 套壳的 WebView 不能正常响应网页中的 `<input type="file">` 文件选择器（点击无反应）。本项目通过实现 `onShowFileChooser` 回调，打通了 WebView 与 Android 系统文件选择器的通道：

- 支持单文件和多文件选择（`<input multiple>`）
- 正确处理旧回调取消逻辑，避免内存泄漏
- 通过 `onActivityResult` 将用户选择的文件 URI 回传给网页

同时在 `AndroidManifest.xml` 中配置了 `FileProvider`，用于安全地向系统共享文件路径：

```xml
<provider
    android:name="androidx.core.content.FileProvider"
    android:authorities="${applicationId}.fileprovider"
    android:exported="false"
    android:grantUriPermissions="true">
    <meta-data
        android:name="android.support.FILE_PROVIDER_PATHS"
        android:resource="@xml/file_paths"/>
</provider>
```

---

### 4. WebView 硬件加速

启用 GPU 硬件加速渲染，改善 WebView 内页面的流畅度，尤其对含动画、视频或复杂布局的 Flask 页面效果明显：

```java
webView.setLayerType(WebView.LAYER_TYPE_HARDWARE, null);
```

---

### 5. 动态权限申请（兼容多 Android 版本）

`requestNecessaryPermissions()` 方法在应用启动时检测并申请必要权限，并针对不同 Android 版本做了区分处理：

| Android 版本 | 申请的存储权限 |
|-------------|-------------|
| Android 13+（API 33+） | `READ_MEDIA_IMAGES` / `READ_MEDIA_VIDEO` / `READ_MEDIA_AUDIO` |
| Android 6 ~ 12（API 23~32） | `READ_EXTERNAL_STORAGE` / `WRITE_EXTERNAL_STORAGE` |
| 所有版本 | `CAMERA` / `RECORD_AUDIO` |

---

## 关键文件详解

### `MainActivity.java`

本项目最核心的文件，继承自 `BridgeActivity`（Capacitor 基类），在 `onCreate` 中对 WebView 进行了全面配置。完整功能包括：SSL 忽略、硬件加速、文件选择器、摄像头/麦克风授权、动态权限申请、权限结果回调、文件选择结果回调。

注意：文件顶部保留了大量注释掉的旧版代码（`// ...`），这是早期迭代版本的记录，当前实际运行的是下方未注释的代码。

### `AndroidManifest.xml`

声明了应用所需的全部权限，关键项包括：

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
<uses-permission android:name="android.permission.READ_MEDIA_VIDEO" />
<uses-permission android:name="android.permission.READ_MEDIA_AUDIO" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.INTERNET" />
```

摄像头和麦克风硬件特性均设置为 `required="false"`，提高设备兼容性（无摄像头设备也可安装）。

### `network_security_config.xml`

网络安全配置文件，主要作用：

- 全局禁止明文 HTTP 流量（`cleartextTrafficPermitted="false"`）
- 为 ngrok 服务器域名单独配置，同时信任系统证书和用户安装的证书
- 为 localhost / 127.0.0.1 / 10.0.2.2 配置本地调试信任

当 Flask 服务器地址变更时，此文件中的 `<domain>` 也需要同步更新。

### `capacitor.config.json`（根目录 & assets 目录各一份）

两份配置内容相同，根目录的是 Capacitor CLI 读取的源配置，`android/app/src/main/assets/` 下的是每次执行 `npx cap sync` 后自动同步过来的副本：

```json
{
  "appId": "com.mycompany.flaskapp",
  "appName": "仪态万象",
  "webDir": "www",
  "server": {
    "url": "https://calamari-scorecard-unsolved.ngrok-free.dev",
    "cleartext": true,
    "androidScheme": "https"
  },
  "android": {
    "allowMixedContent": true
  }
}
```

### `app/build.gradle`

当前版本信息：

```groovy
versionCode 18
versionName "2.8.3"
```

目标 SDK 配置（来自 `variables.gradle`）：

```
minSdkVersion     = 23   (Android 6.0+)
compileSdkVersion = 35
targetSdkVersion  = 35
```

---

## 构建与使用

### 前提条件

- Android Studio（已配置 Android SDK）
- `ANDROID_HOME` 环境变量已设置
- Node.js 14+ 和 npm

### 直接用 Android Studio 打开

```bash
# 进入 android 目录，用 Android Studio 打开
# 或直接双击 android/build.gradle
```

然后在 Android Studio 中选择 **Build → Build APK(s)** 生成 debug APK。

### 命令行构建

```bash
cd android
./gradlew assembleDebug
```

APK 输出路径：

```
android/app/build/outputs/apk/debug/app-debug.apk
```

### 修改服务器后重新同步

如果 Flask 服务器地址发生变化，修改根目录 `capacitor.config.json` 后执行：

```bash
npx cap sync
```

再重新构建 APK。

---

## 修改服务器地址

需要同步修改以下两处：

**1. 根目录 `capacitor.config.json`**（`server.url` 字段）

**2. `android/app/src/main/res/xml/network_security_config.xml`**（`<domain>` 标签）

```xml
<!-- 将旧域名替换为新域名 -->
<domain includeSubdomains="true">你的新域名.ngrok-free.app</domain>
```

修改完毕后运行 `npx cap sync` 同步到 Android 项目，再重新构建。

---

## 权限说明

| 权限 | 用途 | 申请时机 |
|------|------|---------|
| `INTERNET` | 访问 Flask 服务器 | 安装时自动授予 |
| `CAMERA` | 网页调用摄像头 | 首次启动时弹框申请 |
| `RECORD_AUDIO` | 网页录音功能 | 首次启动时弹框申请 |
| `READ_MEDIA_IMAGES/VIDEO/AUDIO` | 文件上传（Android 13+） | 首次启动时弹框申请 |
| `READ/WRITE_EXTERNAL_STORAGE` | 文件上传（Android 6-12） | 首次启动时弹框申请 |
| `MANAGE_EXTERNAL_STORAGE` | 管理外部存储（可选） | 声明但不主动申请 |

---

## 版本信息

| 字段 | 值 |
|------|----|
| 应用名称 | 仪态万象 |
| 包名 | com.mycompany.flaskapp |
| versionCode | 18 |
| versionName | 2.8.3 |
| minSdkVersion | 23（Android 6.0+） |
| targetSdkVersion | 35（Android 15） |
| Capacitor | @capacitor/android（节点依赖） |

---

## 注意事项

- **MainActivity.java 顶部的注释代码**是历史迭代版本，无需删除，实际运行的是文件下半段未注释的代码。
- **SSL 忽略配置**（`handler.proceed()`）适合开发/内网场景，正式上线应替换为正规证书并移除该配置。
- **ngrok 免费地址不固定**，每次重启 ngrok 隧道地址会变更，需同步更新 `capacitor.config.json` 和 `network_security_config.xml` 后重新构建 APK。
- **当前 APK 为 debug 版本**，如需正式发布需在 Android Studio 中进行签名打包（Build → Generate Signed Bundle/APK）。
- `node_modules` 目录体积较大，建议加入 `.gitignore`，通过 `npm install` 重新安装依赖。
