# 微信小程序完整文件清单

## 项目概述
这是一个基于Flask后端的指针式仪表读数自动识别系统微信小程序，使用YOLOv8深度学习模型进行智能检测。

## 文件结构

```
miniprogram/
├── 📄 配置文件
│   ├── app.json                    # 小程序全局配置
│   ├── app.js                      # 小程序入口文件
│   ├── app.wxss                    # 全局样式
│   ├── project.config.json         # 项目配置
│   └── sitemap.json                # 索引配置
│
├── 📄 文档
│   ├── README.md                   # 项目说明文档
│   ├── QUICKSTART.md               # 快速开始指南
│   ├── BACKEND_GUIDE.md            # 后端适配指南
│   ├── DEPLOYMENT.md               # 部署文档
│   └── FILE_LIST.md                # 本文件
│
├── 📁 pages/                       # 页面目录
│   │
│   ├── 📁 login/                   # 登录页面
│   │   ├── login.wxml              # 页面结构
│   │   ├── login.js                # 页面逻辑
│   │   ├── login.wxss              # 页面样式
│   │   └── login.json              # 页面配置（自动生成）
│   │
│   ├── 📁 register/                # 注册页面
│   │   ├── register.wxml
│   │   ├── register.js
│   │   ├── register.wxss
│   │   └── register.json
│   │
│   ├── 📁 index/                   # 首页（检测页面）
│   │   ├── index.wxml              # 主要功能：拍照/上传检测
│   │   ├── index.js                # 检测逻辑、进度显示
│   │   ├── index.wxss
│   │   └── index.json
│   │
│   ├── 📁 history/                 # 历史记录页面
│   │   ├── history.wxml            # 检测历史列表
│   │   ├── history.js              # 分页加载
│   │   ├── history.wxss
│   │   └── history.json
│   │
│   ├── 📁 detail/                  # 详情页面
│   │   ├── detail.wxml             # 检测结果详情
│   │   ├── detail.js               # 修改、确认功能
│   │   ├── detail.wxss
│   │   └── detail.json
│   │
│   ├── 📁 profile/                 # 个人中心页面
│   │   ├── profile.wxml            # 用户信息、设置
│   │   ├── profile.js              # 退出登录、日志查看
│   │   ├── profile.wxss
│   │   └── profile.json
│   │
│   └── 📁 chart/                   # 趋势图表页面
│       ├── chart.wxml              # 序列号历史趋势
│       ├── chart.js                # Canvas绘制图表
│       ├── chart.wxss
│       └── chart.json
│
└── 📁 utils/                       # 工具模块
    ├── request.js                  # 网络请求封装
    ├── api.js                      # API接口定义
    └── util.js                     # 通用工具函数

```

## 文件说明

### 核心配置（必需）

1. **app.json** - 小程序全局配置
   - 页面路由配置
   - TabBar配置
   - 窗口样式
   - 权限声明

2. **app.js** - 小程序入口
   - 全局数据管理
   - 登录状态管理
   - 生命周期函数

3. **app.wxss** - 全局样式
   - 通用样式类
   - CSS变量定义
   - 动画效果

4. **project.config.json** - 项目配置
   - 开发者工具配置
   - 编译选项
   - AppID配置

### 页面文件（7个页面）

每个页面包含4个文件：
- **.wxml** - 页面结构（类似HTML）
- **.wxss** - 页面样式（类似CSS）
- **.js** - 页面逻辑（JavaScript）
- **.json** - 页面配置（可选，一般为空对象{}）

### 工具模块

1. **utils/request.js**
   - HTTP请求封装
   - 上传文件封装
   - 统一错误处理
   - Token管理

2. **utils/api.js**
   - 用户接口（登录、注册、退出）
   - 检测接口（上传、查询、确认、修改）
   - 历史接口（列表、序列号历史）
   - 日志接口（查看、清除）

3. **utils/util.js**
   - 时间格式化
   - 表单验证
   - Toast提示
   - 图片URL处理

### 文档文件

1. **README.md** - 完整项目文档
   - 功能介绍
   - 安装配置
   - API说明
   - 故障排查

2. **QUICKSTART.md** - 5分钟快速开始
   - 最简单的上手步骤
   - 常见问题解答

3. **BACKEND_GUIDE.md** - 后端适配指南
   - Token认证实现
   - 接口修改说明
   - 代码示例

4. **DEPLOYMENT.md** - 生产部署
   - 服务器配置
   - HTTPS配置
   - 性能优化
   - 安全加固

## 缺失的资源文件

⚠️ 以下资源文件需要自行准备：

### images/ 目录（图标图片）

需要准备以下图标文件（建议尺寸：png格式）：

**TabBar图标** (81x81px):
- camera.png / camera-active.png
- history.png / history-active.png
- profile.png / profile-active.png

**页面图标** (40-80rpx):
- logo.png (登录页logo)
- user.png (用户图标)
- lock.png (密码图标)
- eye-open.png / eye-close.png (密码显示切换)
- serial.png (序列号图标)
- upload.png (上传图标)
- camera-btn.png (拍照按钮)
- album.png (相册图标)
- close.png (关闭图标)
- result.png (结果图标)
- task.png (任务图标)
- check.png / check-small.png (确认图标)
- empty.png (空状态图)
- avatar.png (默认头像)
- arrow-right.png (箭头)
- history-icon.png
- log-icon.png
- download-icon.png
- server-icon.png
- version-icon.png

**图标资源获取**：
- [iconfont阿里图标库](https://www.iconfont.cn/)
- [icons8](https://icons8.com/)
- [flaticon](https://www.flaticon.com/)

或使用纯色图标库快速替代：
```bash
# 可以使用emoji转图片，或者用CSS绘制简单图标
# 也可以暂时留空，后续补充
```

## 开始使用

### 最小化启动（可选跳过图标）

1. 创建空的images目录：
```bash
mkdir miniprogram/images
```

2. 暂时注释掉图标：
在各页面的wxml中，将 `<image src="/images/xxx.png">` 暂时替换为文字

3. 修改TabBar配置：
在app.json中暂时注释掉tabBar的iconPath

### 完整启动

1. 准备所有图标文件
2. 放入 `miniprogram/images/` 目录
3. 导入微信开发者工具
4. 开始开发

## 技术栈

- **前端框架**: 微信小程序原生
- **UI设计**: 自定义组件 + 渐变色主题
- **状态管理**: 全局globalData + 页面data
- **网络请求**: wx.request + Promise封装
- **图表绘制**: Canvas API
- **认证方式**: JWT Token

## 功能清单

- [x] 用户登录/注册
- [x] 拍照/上传图片
- [x] YOLO检测（轮询结果）
- [x] 检测进度显示
- [x] 结果查看/修改/确认
- [x] 历史记录（分页）
- [x] 序列号趋势图
- [x] 操作日志
- [x] 个人中心
- [x] Token认证

## 统计信息

- **总文件数**: 32个
- **代码行数**: 约3000+行
- **页面数量**: 7个
- **接口数量**: 12个
- **文档页数**: 4个

## 版本信息

- **版本**: v2.9.3
- **创建日期**: 2025-01-15
- **最低基础库**: 2.19.4
- **开发框架**: 微信小程序

## 授权说明

本项目为教育和学习目的创建。
© 2026 仪态万象. All rights reserved.

---

## 下一步

1. 📖 阅读 [QUICKSTART.md](QUICKSTART.md) 快速开始
2. 🔧 参考 [BACKEND_GUIDE.md](BACKEND_GUIDE.md) 适配后端
3. 🚀 查看 [DEPLOYMENT.md](DEPLOYMENT.md) 部署上线
4. 📚 详细了解 [README.md](README.md) 完整功能

祝开发顺利！🎉
