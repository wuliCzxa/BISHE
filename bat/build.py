# -*- coding: utf-8 -*-
import os
import subprocess
import sys
from pathlib import Path

# ================== 配置区 ==================
APP_NAME = "仪态万象"
APP_ID = "com.mycompany.flaskapp"
# SERVER_URL = "https://192.168.45.79:5000"
SERVER_URL = "https://calamari-scorecard-unsolved.ngrok-free.dev"
PROJECT_DIR = "flask-android-app-ytwx"
# ===========================================


def run(cmd, check=True):
    """执行命令"""
    print(f"\n[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print("[ERROR] 命令执行失败")
        sys.exit(1)


def check_node():
    print("\n[步骤1] 检查 Node.js...")
    try:
        subprocess.run("node --version", shell=True, check=True)
        print("✓ Node.js 已安装")
        run("node --version", check=False)
    except:
        print("[错误] 未找到 Node.js")
        print("请安装：https://nodejs.org/")
        sys.exit(1)

    print("\n检查 npm...")
    try:
        subprocess.run("npm --version", shell=True, check=True)
        print("✓ npm 已安装")
        run("npm --version", check=False)
    except:
        print("[错误] npm 不可用")
        sys.exit(1)


def check_android_home():
    print("\n检查 ANDROID_HOME...")
    android_home = os.environ.get("ANDROID_HOME")

    if not android_home:
        print("[警告] 未检测到 ANDROID_HOME")
        cont = input("是否继续? (y/n): ")
        if cont.lower() != "y":
            sys.exit(0)
    else:
        print(f"✓ ANDROID_HOME = {android_home}")


def create_project():
    print("\n[步骤2] 创建项目目录...")

    p = Path(PROJECT_DIR)
    if p.exists():
        choice = input("目录已存在，是否删除重建? (y/n): ")
        if choice.lower() == "y":
            subprocess.run(f"rmdir /s /q {PROJECT_DIR}", shell=True)
            p.mkdir()
        else:
            print("使用现有目录")
    else:
        p.mkdir()

    os.chdir(PROJECT_DIR)
    print(f"当前目录: {os.getcwd()}")


def init_npm():
    print("\n[步骤3] 初始化 npm...")
    if not Path("package.json").exists():
        run("npm init -y")
    print("✓ npm 初始化完成")


def install_capacitor():
    print("\n[步骤4] 安装 Capacitor...")
    run("npm install @capacitor/core @capacitor/cli @capacitor/android")


#   "server": {{
#     "url": "{SERVER_URL}",
#     "cleartext": true,
#     "androidScheme": "https"
#   }},

def create_capacitor_config():
    print("\n[步骤5] 创建 capacitor.config.json...")

    content = f"""{{
  "appId": "{APP_ID}",
  "appName": "{APP_NAME}",
  "webDir": "www",
  "server": {{
    "url": "https://calamari-scorecard-unsolved.ngrok-free.dev",
    "cleartext": false
  }},
  "android": {{
    "allowMixedContent": true
  }}
}}"""

    with open("capacitor.config.json", "w", encoding="utf-8") as f:
        f.write(content)

    print("✓ 配置文件已创建")


def create_web():
    print("\n[步骤6] 创建 Web 页面...")

    Path("www").mkdir(exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{APP_NAME}</title>
<style>
body,html{{margin:0;height:100%;overflow:hidden;}}
#iframe{{width:100%;height:100%;border:none;}}
.loading{{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);}}
</style>
</head>
<body>

<div class="loading" id="loading">正在连接服务器...</div>

<iframe id="iframe" src="{SERVER_URL}"></iframe>

<script>
const iframe = document.getElementById("iframe");
const loading = document.getElementById("loading");

setTimeout(() => {{
    document.getElementById('error').innerText = "连接超时，请检查服务器";
}}, 10000);
~
iframe.onload = () => loading.style.display="none";
iframe.onerror = () => loading.innerText="连接失败";
</script>

</body>
</html>
"""

    with open("www/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("✓ index.html 创建完成")


def add_android():
    print("\n[步骤7] 添加 Android 平台...")

    if not Path("android").exists():
        run("npx cap add android")
    else:
        print("✓ Android 已存在")


def sync_project():
    print("\n[步骤8] 同步项目...")
    run("npx cap sync")


def build_choose():
    print("\n==============================")
    print("构建方式选择")
    print("1. Android Studio（推荐）")
    print("2. 命令行构建")
    print("==============================")

    choice = input("请选择 (1/2): ")

    if choice == "1":
        print("\n打开 Android Studio...")
        run("npx cap open android", check=False)

    elif choice == "2":
        print("\n命令行构建 APK...")
        os.chdir("android")
        run("gradlew.bat assembleDebug")


def main():
    print("=" * 60)
    print("Flask APK 自动构建脚本 (Python版)")
    print("=" * 60)

    print(f"\n应用名称: {APP_NAME}")
    print(f"应用ID: {APP_ID}")
    print(f"服务器: {SERVER_URL}")

    if input("\n确认配置? (y/n): ").lower() != "y":
        sys.exit(0)

    check_node()
    check_android_home()
    create_project()
    init_npm()
    install_capacitor()
    create_capacitor_config()
    create_web()
    add_android()
    sync_project()
    build_choose()

    print("\n================================================")
    print("项目构建完成")
    print("APK路径:")
    print(r"android\app\build\outputs\apk\debug\app-debug.apk")
    print("================================================")


if __name__ == "__main__":
    main()