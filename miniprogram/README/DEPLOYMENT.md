# 微信小程序部署文档

## 开发环境部署

### 1. 前置条件
- 安装微信开发者工具
- Flask后端服务运行正常
- 数据库已初始化

### 2. 配置步骤

#### 2.1 修改服务器地址
编辑 `app.js`:
```javascript
globalData: {
  serverUrl: 'https://69c1-2408-862e-807-c000-00-68a.ngrok-free.app',  // 改为你的电脑IP
}
```

#### 2.2 开发者工具配置
1. 打开微信开发者工具
2. 导入项目，选择 `miniprogram` 目录
3. AppID选择"测试号"
4. 勾选"不校验合法域名"

#### 2.3 启动服务
```bash
# 启动Flask后端（确保网络可访问）
python app.py
```

#### 2.4 测试
- 点击"编译"
- 使用admin/admin123登录
- 测试拍照检测功能

---

## 生产环境部署

### 1. 域名和证书准备

#### 1.1 域名解析
将域名解析到服务器IP：
```
api.yourdomain.com -> your_server_ip
```

#### 1.2 SSL证书
使用Let's Encrypt免费证书：
```bash
# 安装certbot
sudo apt-get install certbot

# 生成证书
sudo certbot certonly --standalone -d api.yourdomain.com
```

### 2. 服务器配置

#### 2.1 安装Nginx
```bash
sudo apt-get install nginx
```

#### 2.2 配置Nginx
创建 `/etc/nginx/sites-available/miniprogram`:
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket支持（如果需要）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /static {
        alias /var/www/miniprogram/static;
        expires 30d;
    }

    location /uploads {
        alias /var/www/miniprogram/uploads;
        expires 7d;
    }

    location /outputs {
        alias /var/www/miniprogram/outputs;
        expires 7d;
    }
}
```

启用配置：
```bash
sudo ln -s /etc/nginx/sites-available/miniprogram /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 2.3 部署Flask应用

使用Gunicorn：
```bash
# 安装gunicorn
pip install gunicorn

# 创建systemd服务
sudo nano /etc/systemd/system/miniprogram.service
```

内容：
```ini
[Unit]
Description=Miniprogram Flask App
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/miniprogram
Environment="PATH=/var/www/miniprogram/venv/bin"
ExecStart=/var/www/miniprogram/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl start miniprogram
sudo systemctl enable miniprogram
```

### 3. 小程序配置

#### 3.1 修改服务器地址
编辑 `app.js`:
```javascript
globalData: {
  serverUrl: 'https://api.yourdomain.com',
}
```

#### 3.2 微信公众平台配置

登录 [微信公众平台](https://mp.weixin.qq.com/):

1. **服务器域名配置**
   - 开发 -> 开发管理 -> 开发设置 -> 服务器域名
   - request合法域名: `https://api.yourdomain.com`
   - uploadFile合法域名: `https://api.yourdomain.com`
   - downloadFile合法域名: `https://api.yourdomain.com`

2. **业务域名配置**（如果使用web-view）
   - 设置 -> 开发设置 -> 业务域名
   - 添加: `api.yourdomain.com`

#### 3.3 上传代码

在微信开发者工具中：
1. 点击"上传"
2. 填写版本号和备注
3. 上传代码

#### 3.4 提交审核

在微信公众平台：
1. 版本管理 -> 开发版本 -> 提交审核
2. 填写功能页面和类目
3. 等待审核（1-7天）

#### 3.5 发布上线

审核通过后：
1. 版本管理 -> 审核版本 -> 发布
2. 用户即可搜索使用

### 4. 数据库配置

#### 4.1 MySQL优化
```sql
-- 设置最大连接数
SET GLOBAL max_connections = 200;

-- 设置查询缓存
SET GLOBAL query_cache_size = 67108864;
SET GLOBAL query_cache_type = 1;
```

#### 4.2 定期备份
创建备份脚本 `/var/www/scripts/backup.sh`:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/mysql"
DB_NAME="BISHE"
DB_USER="root"
DB_PASS="your_password"

mkdir -p $BACKUP_DIR
mysqldump -u$DB_USER -p$DB_PASS $DB_NAME > $BACKUP_DIR/backup_$DATE.sql
# 删除7天前的备份
find $BACKUP_DIR -name "backup_*.sql" -mtime +7 -delete
```

添加到crontab：
```bash
# 每天凌晨2点备份
0 2 * * * /var/www/scripts/backup.sh
```

### 5. 监控和日志

#### 5.1 配置日志
修改 `app.py`:
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler(
        '/var/log/miniprogram/app.log',
        maxBytes=10240000,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Miniprogram startup')
```

#### 5.2 性能监控
使用Prometheus + Grafana监控：
```bash
# 安装prometheus-flask-exporter
pip install prometheus-flask-exporter

# 在app.py中添加
from prometheus_flask_exporter import PrometheusMetrics
metrics = PrometheusMetrics(app)
```

### 6. 安全加固

#### 6.1 防火墙配置
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

#### 6.2 限制上传文件大小
```python
# app.py
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

#### 6.3 防止SQL注入
使用参数化查询（已实现）

#### 6.4 防止CSRF
```bash
pip install flask-wtf

# 在app.py中
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)
```

### 7. 性能优化

#### 7.1 使用Redis缓存
```bash
pip install redis flask-caching

# app.py
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'redis', 'CACHE_REDIS_URL': 'redis://localhost:6379/0'})
```

#### 7.2 数据库连接池
```python
# 使用pymysql连接池
from dbutils.pooled_db import PooledDB
import pymysql

pool = PooledDB(
    creator=pymysql,
    maxconnections=10,
    mincached=2,
    **DB_CONFIG
)

def get_db():
    return pool.connection()
```

#### 7.3 静态资源CDN
将图片等静态资源上传到CDN

### 8. 故障排查

#### 8.1 查看Nginx日志
```bash
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

#### 8.2 查看Flask日志
```bash
tail -f /var/log/miniprogram/app.log
```

#### 8.3 查看服务状态
```bash
sudo systemctl status miniprogram
sudo systemctl status nginx
sudo systemctl status mysql
```

### 9. 更新部署

#### 9.1 代码更新
```bash
cd /var/www/miniprogram
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart miniprogram
```

#### 9.2 小程序更新
1. 修改代码
2. 上传新版本
3. 提交审核
4. 发布

### 10. 回滚方案

#### 10.1 数据库回滚
```bash
mysql -u root -p BISHE < /var/backups/mysql/backup_20250115_020000.sql
```

#### 10.2 代码回滚
```bash
git reset --hard <commit_hash>
sudo systemctl restart miniprogram
```

#### 10.3 小程序回滚
在微信公众平台版本管理中回退版本

---

## 检查清单

部署前检查：
- [ ] 域名解析正确
- [ ] SSL证书有效
- [ ] 数据库备份
- [ ] 代码测试通过
- [ ] 日志配置正确
- [ ] 监控系统运行

部署后检查：
- [ ] 服务正常运行
- [ ] 小程序可以登录
- [ ] 检测功能正常
- [ ] 图片可以访问
- [ ] 历史记录正常
- [ ] 性能满足要求

## 联系支持

遇到问题请查看文档或联系技术支持。
