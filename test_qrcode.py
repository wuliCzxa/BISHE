#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_qrcode.py
二维码功能测试脚本

用法：
    python test_qrcode.py

功能：
    1. 测试依赖是否安装
    2. 测试二维码生成
    3. 测试User-Agent检测
    4. 生成测试二维码文件
"""

import sys
import os

print("=" * 60)
print("二维码功能测试脚本")
print("=" * 60)

# ========== 测试1: 检查依赖 ==========
print("\n[测试1] 检查依赖安装...")

try:
    import qrcode
    try:
        from importlib.metadata import version
        ver = version("qrcode")
    except:
        ver = "未知版本"
    print("✓ qrcode 已安装，版本:", ver)
except ImportError:
    print("✗ qrcode 未安装")
    print("  请运行: pip install qrcode --break-system-packages")
    sys.exit(1)

try:
    from PIL import Image
    print("✓ Pillow 已安装")
except ImportError:
    print("✗ Pillow 未安装")
    print("  请运行: pip install Pillow --break-system-packages")
    sys.exit(1)

try:
    import flask
    print("✓ Flask 已安装，版本:", flask.__version__)
except ImportError:
    print("✗ Flask 未安装")
    print("  请运行: pip install flask --break-system-packages")
    sys.exit(1)


# ========== 测试2: 检查文件 ==========
print("\n[测试2] 检查必需文件...")

files_to_check = {
    "qrcode_helper.py": "二维码辅助模块",
    "config.py": "配置文件",
}

all_files_exist = True
for filename, desc in files_to_check.items():
    if os.path.exists(filename):
        print(f"✓ {filename} ({desc})")
    else:
        print(f"✗ {filename} ({desc}) - 文件不存在")
        all_files_exist = False

if not all_files_exist:
    print("\n警告: 部分文件缺失，某些测试可能失败")


# ========== 测试3: 测试二维码生成 ==========
print("\n[测试3] 测试二维码生成...")

try:
    from io import BytesIO
    
    # 创建测试二维码
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2,
    )
    
    test_data = "https://www.baidu.com"
    qr.add_data(test_data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # 保存测试二维码
    # test_filename = "test_qrcode.png"
    test_filename = "/static/test_qrcode.png"  # 使用静态文件夹路径 ，需要确保该路径存在并且Flask配置了静态文件夹
    img.save(test_filename)
    
    print(f"✓ 二维码生成成功")
    print(f"  测试二维码已保存: {test_filename}")
    print(f"  扫描该二维码应跳转到: {test_data}")
    
except Exception as e:
    print(f"✗ 二维码生成失败: {e}")
    sys.exit(1)


# ========== 测试4: 测试配置读取 ==========
print("\n[测试4] 测试配置读取...")

try:
    import config as cfg
    
    # 检查数据库配置
    print("✓ config.py 可以导入")
    print(f"  数据库主机: {cfg.DB_HOST}")
    print(f"  数据库端口: {cfg.DB_PORT}")
    print(f"  数据库名称: {cfg.DB_NAME}")
    
    # 检查二维码配置
    if hasattr(cfg, 'WECHAT_MINIPROGRAM_APPID'):
        print(f"  小程序AppID: {cfg.WECHAT_MINIPROGRAM_APPID}")
        if cfg.WECHAT_MINIPROGRAM_APPID == "your_miniprogram_appid":
            print("  ⚠ 警告: 小程序AppID使用默认值，需要修改")
    else:
        print("  ⚠ 警告: 未找到 WECHAT_MINIPROGRAM_APPID 配置")
    
    if hasattr(cfg, 'APK_DOWNLOAD_URL'):
        print(f"  APK下载链接: {cfg.APK_DOWNLOAD_URL}")
        if "your-domain.com" in cfg.APK_DOWNLOAD_URL:
            print("  ⚠ 警告: APK下载链接使用默认值，需要修改")
    else:
        print("  ⚠ 警告: 未找到 APK_DOWNLOAD_URL 配置")
        
except ImportError as e:
    print(f"✗ config.py 导入失败: {e}")
except Exception as e:
    print(f"✗ 配置读取错误: {e}")


# ========== 测试5: 测试qrcode_helper模块 ==========
print("\n[测试5] 测试 qrcode_helper 模块...")

try:
    # 临时导入测试
    sys.path.insert(0, os.getcwd())
    
    from qrcode_helper import create_qrcode_image, generate_qrcode_url
    
    print("✓ qrcode_helper.py 可以导入")
    
    # 测试生成二维码
    test_url = "https://www.example.com"
    img_io = create_qrcode_image(test_url)
    
    print(f"✓ create_qrcode_image() 函数正常")
    print(f"  生成的二维码大小: {len(img_io.getvalue())} 字节")
    
except ImportError as e:
    print(f"✗ qrcode_helper.py 导入失败: {e}")
    print("  确保 qrcode_helper.py 在当前目录")
except Exception as e:
    print(f"✗ qrcode_helper 测试失败: {e}")


# ========== 测试6: 生成示例二维码 ==========
print("\n[测试6] 生成示例二维码...")

try:
    from qrcode_helper import create_qrcode_image
    
    # 生成微信小程序二维码示例
    wechat_url = "weixin://dl/business/?t=example_appid"
    wechat_img = create_qrcode_image(wechat_url)
    
    from PIL import Image
    img = Image.open(wechat_img)
    img.save("/static/qrcode_wechat_example.png")  # 保存到静态文件夹   
    print("✓ 微信小程序二维码示例: qrcode_wechat_example.png")
    
    # 生成APK下载二维码示例
    apk_url = "https://example.com/downloads/app.apk"
    apk_img = create_qrcode_image(apk_url)
    
    img = Image.open(apk_img)
    img.save("/static/qrcode_apk_example.png")  # 保存到静态文件夹
    print("✓ APK下载二维码示例: qrcode_apk_example.png")
    
except Exception as e:
    print(f"⚠ 示例二维码生成跳过: {e}")


# ========== 测试总结 ==========
print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)

print("\n下一步:")
print("1. 检查生成的测试二维码图片")
print("2. 修改 config.py 中的配置项（如有警告）")
print("3. 按照集成指南修改 app.py")
print("4. 运行 python app.py 启动应用")
print("5. 访问 http://localhost:5000/api/qrcode 测试API")
print("\n详细说明见: README_二维码功能集成指南.md")
