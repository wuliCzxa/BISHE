#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup_apk_download.py
设置 APK 下载文件夹结构

运行：python setup_apk_download.py
"""

import os

# 创建必要的文件夹
folders = [
    "./static",
    "./static/downloads",
]

print("创建文件夹结构...")
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✓ {folder}")

print("\n文件夹结构创建完成！")
print("\n下一步：")
print("1. 将编译好的 APK 复制到: ./static/downloads/app.apk")
print("   源文件: android/app/build/outputs/apk/debug/app-debug.apk")
print("2. 运行: python app.py")
print("3. 访问: http://localhost:5000/api/qrcode 测试二维码")
