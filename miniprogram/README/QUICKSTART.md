# 快速开始指南

## 5分钟快速体验

### 1. 准备工作（2分钟）

#### 下载工具
1. 下载并安装[微信开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)
2. 确保Flask后端正在运行

#### 检查后端
```bash
# 确认Flask服务运行在http://localhost:5000
curl http://localhost:5000/api/qrcode/info
```

### 2. 导入项目（1分钟）

1. 打开微信开发者工具
2. 点击"导入项目"
3. 选择 `miniprogram` 目录
4. AppID选择"测试号"（或填入你的AppID）
5. 点击"导入"

### 3. 配置服务器（1分钟）

编辑 `app.js`，修改第3行：
```javascript
serverUrl: 'http://localhost:5000',  // 改为你的后端地址
```

如果后端在其他电脑：
```javascript
serverUrl: 'https://69c1-2408-862e-807-c000-00-68a.ngrok-free.app',  // IP改为你的电脑IP
```

### 4. 开启调试模式（30秒）

点击右上角"详情"，勾选：
- ☑️ 不校验合法域名、web-view（业务域名）、TLS版本以及HTTPS证书
- ☑️ 启用ES6转ES5
- ☑️ 启用增强编译

### 5. 开始使用（30秒）

1. 点击"编译"
2. 使用默认账号登录：
   - 用户名：`admin`
   - 密码：`admin123`
3. 拍照或选择图片进行检测！

---

## 常见问题

### Q: 提示"网络请求失败"
**A:** 检查：
1. Flask后端是否运行
2. serverUrl地址是否正确
3. 是否勾选"不校验合法域名"

### Q: 无法登录
**A:** 检查：
1. 数据库是否已初始化（运行check_db.py）
2. 用户名密码是否正确
3. 查看后端控制台错误信息

### Q: 图片无法显示
**A:** 检查：
1. 后端/image接口是否正常
2. 图片路径是否正确
3. 文件夹权限是否正确

### Q: 编译报错
**A:** 
1. 检查app.json语法是否正确
2. 确认所有页面文件都存在
3. 重新编译或重启开发者工具

---

## 下一步

- 阅读 [README.md](README.md) 了解完整功能
- 参考 [BACKEND_GUIDE.md](BACKEND_GUIDE.md) 适配后端
- 查看 [DEPLOYMENT.md](DEPLOYMENT.md) 部署到生产环境

---

## 需要帮助？

1. 查看微信小程序[官方文档](https://developers.weixin.qq.com/miniprogram/dev/framework/)
2. 检查后端日志
3. 使用微信开发者工具的调试器

祝使用愉快！🎉
