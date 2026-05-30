#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_miniprogram.py
一键生成微信小程序完整项目

运行方式：python create_miniprogram.py
生成的项目在：./miniprogram/ 目录
"""

import os
import json

# 项目基础目录
BASE_DIR = './miniprogram'

# 创建目录结构
DIRS = [
    'pages/login',
    'pages/index',
    'pages/result',
    'pages/history',
    'pages/profile',
    'pages/serial-chart',
    'utils',
    'components/loading',
    'images'
]

# 页面配置模板
PAGE_JSON_TEMPLATE = {
    "navigationBarTitleText": "",
    "enablePullDownRefresh": False
}

print("=" * 60)
print("微信小程序项目生成工具")
print("=" * 60)

# 步骤 1: 创建目录
print("\n[步骤1] 创建目录结构...")
for dir_path in DIRS:
    full_path = os.path.join(BASE_DIR, dir_path)
    os.makedirs(full_path, exist_ok=True)
    print(f"✓ {dir_path}")

# 步骤 2: 创建页面配置文件
print("\n[步骤2] 创建页面配置文件...")

pages_config = {
    'login': {'navigationBarTitleText': '登录', 'navigationBarBackgroundColor': '#667eea', 'navigationBarTextStyle': 'white'},
    'index': {'navigationBarTitleText': '仪表检测', 'enablePullDownRefresh': True},
    'result': {'navigationBarTitleText': '检测结果', 'enablePullDownRefresh': False},
    'history': {'navigationBarTitleText': '历史记录', 'enablePullDownRefresh': True},
    'profile': {'navigationBarTitleText': '个人中心', 'enablePullDownRefresh': False},
    'serial-chart': {'navigationBarTitleText': '历史趋势', 'enablePullDownRefresh': False}
}

for page_name, config in pages_config.items():
    json_path = os.path.join(BASE_DIR, 'pages', page_name, f'{page_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"✓ pages/{page_name}/{page_name}.json")

# 步骤 3: 创建sitemap.json
print("\n[步骤3] 创建配置文件...")
sitemap = {
    "desc": "关于本文件的更多信息，请参考文档 https://developers.weixin.qq.com/miniprogram/dev/framework/sitemap.html",
    "rules": [{
        "action": "allow",
        "page": "*"
    }]
}

sitemap_path = os.path.join(BASE_DIR, 'sitemap.json')
with open(sitemap_path, 'w', encoding='utf-8') as f:
    json.dump(sitemap, f, ensure_ascii=False, indent=2)
print("✓ sitemap.json")

# 步骤 4: 创建 project.config.json
project_config = {
    "description": "仪表读数检测系统",
    "packOptions": {
        "ignore": [],
        "include": []
    },
    "setting": {
        "bundle": False,
        "userConfirmedBundleSwitch": False,
        "urlCheck": False,
        "scopeDataCheck": False,
        "coverView": True,
        "es6": True,
        "postcss": True,
        "compileHotReLoad": True,
        "lazyloadPlaceholderEnable": False,
        "preloadBackgroundData": False,
        "minified": True,
        "autoAudits": False,
        "newFeature": False,
        "uglifyFileName": False,
        "uploadWithSourceMap": True,
        "useIsolateContext": True,
        "nodeModules": False,
        "enhance": True,
        "useMultiFrameRuntime": True,
        "useApiHook": True,
        "useApiHostProcess": True,
        "showShadowRootInWxmlPanel": True,
        "packNpmManually": False,
        "enableEngineNative": False,
        "packNpmRelationList": [],
        "minifyWXSS": True,
        "showES6CompileOption": False,
        "minifyWXML": True,
        "babelSetting": {
            "ignore": [],
            "disablePlugins": [],
            "outputPath": ""
        }
    },
    "compileType": "miniprogram",
    "libVersion": "2.19.4",
    "appid": "touristappid",
    "projectname": "仪表读数检测系统",
    "condition": {}
}

project_config_path = os.path.join(BASE_DIR, 'project.config.json')
with open(project_config_path, 'w', encoding='utf-8') as f:
    json.dump(project_config, f, ensure_ascii=False, indent=2)
print("✓ project.config.json")

# 完成
print("\n" + "=" * 60)
print("项目结构创建完成！")
print("=" * 60)

print("\n📁 项目目录: " + os.path.abspath(BASE_DIR))

print("\n📋 下一步:")
print("1. 将以下文件复制到对应目录:")
print("   - app.js, app.json, app.wxss → miniprogram/")
print("   - api.js, util.js → miniprogram/utils/")
print("   - login.wxml, login.js, login.wxss → miniprogram/pages/login/")
print("")
print("2. 修改 app.js 中的服务器地址:")
print("   apiBase: 'https://your-domain.com:5000'")
print("")
print("3. 准备图标文件（放在 miniprogram/images/）:")
print("   - logo.png")
print("   - icon-upload.png / icon-upload-active.png")
print("   - icon-history.png / icon-history-active.png")
print("   - icon-profile.png / icon-profile-active.png")
print("")
print("4. 用微信开发者工具打开 miniprogram/ 目录")
print("")
print("详细说明见: miniprogram-deployment-guide.md")
