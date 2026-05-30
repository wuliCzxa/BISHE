#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
deploy_apk_download.py
一键部署 APK 下载功能

功能：
1. 创建必要的文件夹结构
2. 复制 APK 文件到指定位置
3. 验证文件完整性
4. 提供测试链接

运行：python deploy_apk_download.py
"""

import os
import shutil
import sys

print("=" * 60)
print("APK 下载功能部署脚本")
print("=" * 60)

# ========== 步骤1: 创建文件夹 ==========
print("\n[步骤1] 创建文件夹结构...")

folders = [
    "./static",
    "./static/downloads",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✓ {folder}")

# ========== 步骤2: 查找并复制 APK ==========
print("\n[步骤2] 查找 APK 文件...")

# 可能的 APK 源路径
apk_source_paths = [
    # Android Studio 编译输出（相对于 Flask 项目根目录）
    "../flask-android-app/android/app/build/outputs/apk/debug/app-debug.apk",
    "../../BISHE-004/flask-android-app/android/app/build/outputs/apk/debug/app-debug.apk",
    
    # 绝对路径示例（根据你的实际路径修改）
    "D:/BIYESHEJI/BISHE-004/flask-android-app/android/app/build/outputs/apk/debug/app-debug.apk",
]

# 目标路径
apk_target = "./static/downloads/app.apk"

apk_found = False
apk_source = None

# 尝试找到 APK 文件
for source_path in apk_source_paths:
    if os.path.exists(source_path):
        apk_source = source_path
        apk_found = True
        print(f"✓ 找到 APK: {source_path}")
        break

if apk_found:
    # 复制 APK 文件
    try:
        shutil.copy2(apk_source, apk_target)
        print(f"✓ APK 已复制到: {apk_target}")
        
        # 显示文件大小
        size_mb = os.path.getsize(apk_target) / (1024 * 1024)
        print(f"  文件大小: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"✗ 复制失败: {e}")
        sys.exit(1)
else:
    print("⚠ 未找到 APK 文件")
    print("\n请手动复制 APK 文件：")
    print("  源文件: android/app/build/outputs/apk/debug/app-debug.apk")
    print(f"  目标位置: {apk_target}")
    print("\n或者修改此脚本中的 apk_source_paths 变量，添加你的 APK 路径")

# ========== 步骤3: 验证文件 ==========
print("\n[步骤3] 验证文件...")

if os.path.exists(apk_target):
    print(f"✓ APK 文件存在: {apk_target}")
    size_mb = os.path.getsize(apk_target) / (1024 * 1024)
    
    if size_mb < 0.5:
        print(f"⚠ 警告: 文件大小异常 ({size_mb:.2f} MB)，可能不是有效的 APK")
    else:
        print(f"✓ 文件大小正常: {size_mb:.2f} MB")
else:
    print(f"✗ APK 文件不存在: {apk_target}")
    print("  请手动复制 APK 文件到此位置")

# ========== 步骤4: 检查依赖 ==========
print("\n[步骤4] 检查依赖...")

try:
    import qrcode
    print("✓ qrcode 已安装")
except ImportError:
    print("✗ qrcode 未安装")
    print("  运行: pip install qrcode pillow --break-system-packages")

try:
    import flask
    print(f"✓ Flask 已安装 (版本 {flask.__version__})")
except ImportError:
    print("✗ Flask 未安装")
    print("  运行: pip install flask --break-system-packages")

# ========== 步骤5: 显示使用说明 ==========
print("\n" + "=" * 60)
print("部署完成！")
print("=" * 60)

print("\n📋 文件结构:")
print("  项目根目录/")
print("  ├── app.py")
print("  ├── qrcode_helper.py")
print("  └── static/")
print("      └── downloads/")
print(f"          └── app.apk  {'✓ 已就位' if os.path.exists(apk_target) else '✗ 缺失'}")

print("\n🚀 使用方法:")
print("1. 确保已修改 app.py 和 qrcode_helper.py（见说明文档）")
print("2. 运行 Flask 服务器:")
print("   python app.py")
print("\n3. 测试下载:")
print("   浏览器访问: http://localhost:5000/downloads/app.apk")
print("\n4. 测试二维码:")
print("   浏览器访问: http://localhost:5000/api/qrcode")
print("   手机扫描二维码即可下载 APK")

print("\n📱 局域网访问:")
print("   1. 查看本机 IP: ipconfig (Windows) 或 ifconfig (Linux/Mac)")
print("   2. 手机访问: https://<本机IP>:5000/api/qrcode")
print("   3. 扫描生成的二维码下载 APK")

print("\n⚠ 注意事项:")
print("   - HTTPS 模式下会有证书警告，点击'继续访问'即可")
print("   - 确保手机和电脑在同一局域网")
print("   - 防火墙可能需要允许 5000 端口")

if not os.path.exists(apk_target):
    print("\n❌ 警告: APK 文件缺失!")
    print("   请将 APK 文件复制到:")
    print(f"   {os.path.abspath(apk_target)}")
