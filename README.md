# 指针型仪表自动识别系统

> 基于 YOLOv11-OBB（旋转目标检测）+ Flask + MySQL 的工业仪表智能读数平台
>
> 支持 **Web 浏览器端**、**PyQt5 桌面端**、**批量脚本** 三种使用方式，可一键完成表盘定位→标签识别→刻度检测→数值计算的完整读数流程，并提供历史记录查询与多日期趋势图功能。

---

## 目录

- [一、项目简介](#一项目简介)
- [二、系统架构与工作原理](#二系统架构与工作原理)
- [三、目录与文件说明](#三目录与文件说明)
- [四、数据库表结构](#四数据库表结构)
- [五、环境要求](#五环境要求)
- [六、完整部署流程（新手必读）](#六完整部署流程新手必读)
- [七、各功能模块详细使用说明](#七各功能模块详细使用说明)
- [八、模型训练完整流程（开发者）](#八模型训练完整流程开发者)
- [九、核心读数算法详解](#九核心读数算法详解)
- [十、API 接口说明](#十api-接口说明)
- [十一、用户权限体系](#十一用户权限体系)
- [十二、输出文件说明](#十二输出文件说明)
- [十三、常见问题与排错指南](#十三常见问题与排错指南)
- [十四、注意事项与已知限制](#十四注意事项与已知限制)

---

## 一、项目简介

工业现场有大量指针式压力表需要定期人工抄表，本系统利用深度学习替代人工读数：

- **自动化程度高**：拍一张照片，系统全程自动完成四个阶段的图像处理，输出数值读数
- **多端支持**：提供 Web 浏览器界面（可局域网访问、可调用摄像头）和 PyQt5 桌面独立程序两套入口
- **数据可追溯**：所有检测记录入库（MySQL），支持按表计编号查询历史读数变化趋势
- **人工核验**：系统输出结果后，操作员可"确认"或手动"修改"读数，保证数据准确
- **批量处理**：支持一次对整个文件夹的图片批量读数，并将汇总结果导出为 Excel 表格

---

## 二、系统架构与工作原理

### 2.1 整体处理流程

```
┌──────────────────────────────────────────────────────────────┐
│                       原始压力表图片                           │
└─────────────────────────────┬────────────────────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Model 1（YOLOv11） │
                   │  整体仪表区域检测    │
                   │  类别：Instrument    │
                   └──────────┬──────────┘
                              │ 裁剪出「仪表盘 + 标签」区域
              ┌───────────────┴──────────────┐
              │                              │
   ┌──────────▼──────────┐      ┌────────────▼────────────┐
   │  Model 2（YOLOv11） │      │  Model 3（YOLOv11）     │
   │  纯表盘检测（无标签）│      │  仪表编号标签检测        │
   │  类别：Pointer       │      │  类别：Label             │
   └──────────┬──────────┘      └────────────┬────────────┘
              │ 裁剪出纯表盘图                │ 裁剪出标签图
              │                              │
   ┌──────────▼──────────┐      ┌────────────▼────────────┐
   │  Model 4（YOLOv11   │      │  Model 4（YOLOv11       │
   │  OBB 旋转检测）      │      │  OBB 旋转检测）          │
   │  检测刻度线与指针    │      │  识别标签上的序号数字    │
   └──────────┬──────────┘      └────────────┬────────────┘
              │ 返回 OBB 坐标                 │ 查对照表
              │                              │ 得到表计名称
   ┌──────────▼──────────────────────────────▼────────────┐
   │                  角度比例法计算读数                    │
   │  输入：Scale（起始刻度）、Scale2（终止刻度）、         │
   │        中间参考刻度、Pointer（指针）的 OBB 坐标        │
   │  输出：修正前读数（reading_before）                    │
   │        修正后读数（reading_after）                    │
   └──────────────────────────┬───────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
   ┌──────▼──────┐   ┌────────▼───────┐   ┌──────▼──────┐
   │  Web 界面   │   │  MySQL 数据库   │   │  TXT 日志   │
   │  展示结果   │   │  永久存储记录   │   │  追加写入   │
   └─────────────┘   └────────────────┘   └─────────────┘
```

### 2.2 四个模型分工

| 编号 | 模型用途 | 检测类别 | 权重路径（相对 `weight/`） |
|------|---------|---------|--------------------------|
| Model 1 | 从原图检测并裁剪仪表区域（含编号标签） | `Instrument` | `1biaopan_all/weights/best.pt` |
| Model 2 | 从仪表区域进一步裁剪纯表盘（无标签） | `Pointer` | `2biaopan_nolabel/weights/best.pt` |
| Model 3 | 从仪表区域裁剪出编号标签区域 | `Label` | `3biaopan_label/weights/best.pt` |
| Model 4 | OBB 旋转检测，同时承担两个任务：①在标签图上识别序号数字；②在表盘图上检测刻度线与指针 | `Scale/Scale2/Pointer/1~52` | `4read/weights/best.pt` |

### 2.3 Web 服务端架构

```
浏览器 (index.html / login.html)
    │  HTTP(S) 请求
    ▼
Flask (app.py, port 5000)
    ├── 路由层：/login  /index  /upload  /poll/<task_id>
    │           /confirm  /modify  /clear  /get_log
    │           /api/history  /api/serial_history  /image/<path>
    ├── 认证层：Flask Session + werkzeug pbkdf2 密码哈希
    │           login_required 装饰器保护所有核心接口
    ├── 任务层：每次检测在独立后台线程运行（threading.Thread）
    │           状态写入内存字典 _task_states（线程安全）
    │           前端通过轮询 /poll/<task_id> 实时获取进度与结果
    └── 数据层：pymysql 连接 MySQL BISHE 库（user 表 + yolo 表）
```

---

## 三、目录与文件说明

部署后的完整目录结构如下（`*` 标注的目录由程序运行时自动创建）：

```
项目根目录/
│
├── app.py                          # Flask Web 主程序（核心入口）
├── config.py                       # 统一配置文件（数据库/路径/端口）
├── check_db.py                     # 数据库连接诊断与自动修复工具
│
├── pyqt_pressure_HW_yolo_obb.py    # PyQt5 桌面端：单张图片读数（GUI）
├── reading_batch.py                # 批量图片读数脚本（无 GUI）
│
├── dataset_train.py                # Model 1/2/3 训练入口
├── z_1_xml2data.py                 # 标注转换：RoLabelImg XML → DOTA XML + TXT
├── z_2_txt2txt.py                  # 标注转换：DOTA TXT → YOLO-OBB 归一化 TXT
├── z_3train_read.py                # Model 4（OBB 读数模型）训练入口
├── z_4_test_read.py                # Model 4 推理测试脚本
│
├── pic_all.py                      # 生成各表计独立趋势图（每表计一张图）
├── pic_one.py                      # 生成所有表计合并趋势图（一张图）
│
├── 序号标记对照表.xlsx               # 仪表编号序号 → 表计名称对照字典（必须存在）
├── requirements.txt                # 开发环境完整依赖列表（仅供参考）
├── 说明.txt                         # 各脚本功能简要中文说明
│
├── templates/
│   ├── index.html                  # Web 主页（上传检测 + 历史记录 + 趋势图）
│   └── login.html                  # Web 登录页
│
├── uploads/ *                      # 用户上传图片临时存放目录
├── outputs/ *                      # 检测结果输出根目录
│   └── outputs-YYYY-MM-DD/ *       # 按日期分组的子目录
│       └── <任务ID>/               # 每次检测的所有中间图和结果图
│
├── data_show/                      # 批量读数输入图片目录（手动创建，放入待读数图片）
├── a_reading_batch_results/ *      # 批量读数汇总 Excel 永久存储目录
├── a_reading_results_diff_day/     # 趋势图输入：多日期读数 Excel（手动创建）
├── a_reading_results_show_pic_one/ # 趋势图输入：pic_one 读数 Excel（手动创建）
├── a_pic_results_all/ *            # pic_all.py 输出的各表计趋势图
└── a_pic_results_one/ *            # pic_one.py 输出的合并趋势图
│
└── ../ultralytics001/yolo_obb/     # 模型权重与日志（路径可在 config.py 修改）
    ├── 序号标记对照表.xlsx
    ├── Result_pointer.txt           # 读数历史日志（追加写入）
    └── weight/
        ├── 1biaopan_all/weights/best.pt
        ├── 2biaopan_nolabel/weights/best.pt
        ├── 3biaopan_label/weights/best.pt
        └── 4read/weights/best.pt
```

---

## 四、数据库表结构

系统使用 MySQL 数据库 `BISHE`，包含两张表，首次运行 `app.py` 或 `check_db.py` 时**自动创建，无需手动建库建表**。

### 4.1 user 表（用户账号表）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT UNSIGNED, AUTO_INCREMENT, PK | 用户主键 |
| `username` | VARCHAR(64), UNIQUE | 用户名（全局唯一） |
| `password` | VARCHAR(255) | werkzeug pbkdf2_sha256 哈希密码 |
| `user_level` | ENUM('super_admin','admin','user') | 权限级别，默认 'user' |
| `created_at` | DATETIME, DEFAULT CURRENT_TIMESTAMP | 账号创建时间 |

### 4.2 yolo 表（检测记录表）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT UNSIGNED, AUTO_INCREMENT, PK | 记录主键 |
| `task_id` | VARCHAR(32), UNIQUE | 任务 ID，格式：`YYYY-MM-DD-N`（当天第 N 次） |
| `user_id` | INT UNSIGNED, FK→user.id | 操作用户（级联删除） |
| `serial_number` | VARCHAR(64) | 识别到的表计名称（查对照表得到） |
| `original_img_path` | VARCHAR(512) | 原始上传图片的存储路径 |
| `dial_img_path` | VARCHAR(512) | 裁剪后纯表盘图路径 |
| `label_img_path` | VARCHAR(512) | 裁剪后标签区域图路径 |
| `obb_img_path` | VARCHAR(512) | 带标签仪表盘图路径（Model 1 输出） |
| `reading_before` | DECIMAL(12,6) | 修正前读数（圆心估算 2） |
| `reading_after` | DECIMAL(12,6) | 最终读数（双圆心均值修正后） |
| `detect_status` | ENUM('pending','running','success','failed') | 检测状态，默认 'pending' |
| `is_confirmed` | TINYINT(1), DEFAULT 0 | 是否已人工确认（0=未确认，1=已确认） |
| `confirmed_at` | DATETIME | 人工确认时间 |
| `detected_at` | DATETIME | 检测完成时间 |
| `created_at` | DATETIME, DEFAULT CURRENT_TIMESTAMP | 记录创建时间 |

---

## 五、环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Windows 10 / Linux / macOS | Windows 10/11（开发与测试主要平台） |
| Python | 3.10 | 3.10（与 PyQt5 / PyTorch 兼容性最佳） |
| GPU / CUDA | CPU 也可运行，但推理速度慢 | NVIDIA GPU + CUDA 11.8 |
| MySQL | 8.0 | 8.0 |
| 内存（RAM） | 8 GB | 16 GB |
| 磁盘空间 | 4 GB（模型权重约 50\~200 MB × 4） | 20 GB（含训练数据集） |
| 网络 | 仅本地访问时不需要 | 局域网访问时需要确保端口 5000 畅通 |

---

## 六、完整部署流程（新手必读）

请严格按顺序执行以下各步骤。

---

### Step 1 获取项目文件

将全部项目文件解压或克隆到本地目录，例如：

```
D:\bishe\
```

确认目录下包含 `app.py`、`config.py`、`check_db.py`、`templates/` 等核心文件。

---

### Step 2 安装 Python 与虚拟环境

**推荐使用 Conda 隔离环境**，避免与系统其他 Python 项目冲突。

```bash
# 若尚未安装 Conda，前往以下地址下载 Miniconda：
# https://docs.conda.io/en/latest/miniconda.html

# 创建专用虚拟环境
conda create -n bishe python=3.10 -y

# 激活环境（此后所有命令均在此环境下执行）
conda activate bishe

# 验证 Python 版本
python --version    # 应输出 Python 3.10.x
```

---

### Step 3 安装 Python 依赖

> **重要提示**：`requirements.txt` 中部分包记录的是开发机器上的本地 Conda 绝对路径，**不可直接** `pip install -r requirements.txt`。请按以下命令手动安装。

**① 核心依赖（必装）**

```bash
pip install flask pymysql pandas openpyxl numpy opencv-python werkzeug ultralytics
```

**② PyTorch（根据是否有 NVIDIA GPU 二选一）**

```bash
# 有 NVIDIA GPU（CUDA 11.8 对应）：
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 仅 CPU（速度较慢，适合无 GPU 的环境或功能调试）：
pip install torch torchvision
```

**③ PyQt5 桌面端（仅使用桌面程序时需要）**

```bash
pip install PyQt5
```

**④ HTTPS 局域网摄像头支持（可选，但强烈推荐）**

```bash
pip install pyOpenSSL
```

安装后，下次启动 `app.py` 会自动启用 HTTPS 模式，局域网内其他设备（手机、平板）也可通过浏览器调用摄像头拍照上传。

**⑤ 趋势图生成（使用 pic_all.py / pic_one.py 时需要）**

```bash
pip install matplotlib
```

**⑥ 验证安装成功**

```bash
python -c "import flask, pymysql, ultralytics, cv2, numpy; print('核心依赖安装成功 ✔')"
```

---

### Step 4 配置 config.py

用任意文本编辑器打开项目根目录下的 `config.py`，这是**整个系统唯一需要手动编辑的配置文件**，修改后重启 `app.py` 即可生效。

```python
# ============================================================
#  config.py — BISHE 项目统一配置文件
# ============================================================

# ── MySQL 数据库连接 ──────────────────────────────────────
DB_HOST     = "localhost"   # 数据库服务器地址（本机部署填 localhost 或 127.0.0.1）
DB_PORT     = 3306          # MySQL 端口，默认 3306，通常无需修改
DB_USER     = "root"        # MySQL 用户名
DB_PASSWORD = "root"        # ← 【必改】填写你的 MySQL 密码
                             #   XAMPP 默认 root 密码为空字符串：DB_PASSWORD = ""
DB_NAME     = "BISHE"       # 数据库名，程序自动创建，无需提前手动建库

# ── 模型权重与文件路径 ──────────────────────────────────
# 默认假设 ultralytics001 文件夹与项目根目录在同级目录下。
# 如果模型权重放在其他位置，改为绝对路径，例如：
# MODEL_PATH1 = r"D:\models\1biaopan_all\weights\best.pt"

FILE_EXCEL_PATH = "../ultralytics001/yolo_obb/序号标记对照表.xlsx"
MODEL_PATH1 = "../ultralytics001/yolo_obb/weight/1biaopan_all/weights/best.pt"
MODEL_PATH2 = "../ultralytics001/yolo_obb/weight/2biaopan_nolabel/weights/best.pt"
MODEL_PATH3 = "../ultralytics001/yolo_obb/weight/3biaopan_label/weights/best.pt"
MODEL_PATH4 = "../ultralytics001/yolo_obb/weight/4read/weights/best.pt"
TXT_LOG_PATH = "../ultralytics001/yolo_obb/Result_pointer.txt"

# ── 上传 / 输出目录（相对路径，运行时自动创建）──────────
UPLOAD_FOLDER = "uploads"   # 上传图片存放目录
OUTPUT_FOLDER = "outputs"   # 检测结果输出目录

# ── Flask 服务设置 ────────────────────────────────────
FLASK_SECRET_KEY = "bishe_flask_secret_2025"   # Session 加密密钥（可随意改）
FLASK_HOST = "0.0.0.0"     # 监听所有网卡，允许局域网访问
FLASK_PORT = 5000           # Web 服务端口（默认 5000）
FLASK_DEBUG = True          # 调试模式，生产部署时改为 False
```

**常见部署场景对应配置速查：**

| 部署场景 | `DB_HOST` | `DB_PASSWORD` |
|---------|-----------|---------------|
| XAMPP（本机） | `"localhost"` | `""` （空字符串） |
| MySQL Workbench / MySQL Installer（本机） | `"localhost"` | 安装时自设的密码 |
| 宝塔面板（本机） | `"127.0.0.1"` | 宝塔数据库密码 |
| 云服务器 / 远程 MySQL | 服务器公网 IP | 对应 MySQL 账号密码 |

---

### Step 5 准备模型权重文件

将训练好的 4 个 `.pt` 权重文件放置到 `config.py` 所指定的路径下。使用默认路径时，目录结构如下：

```
D:\                              ← 磁盘根目录（或任意位置）
├── bishe\                       ← 项目根目录（app.py 所在处）
└── ultralytics001\
    └── yolo_obb\
        ├── 序号标记对照表.xlsx
        ├── Result_pointer.txt   （首次运行时自动创建）
        └── weight\
            ├── 1biaopan_all\
            │   └── weights\
            │       └── best.pt   ← Model 1 权重
            ├── 2biaopan_nolabel\
            │   └── weights\
            │       └── best.pt   ← Model 2 权重
            ├── 3biaopan_label\
            │   └── weights\
            │       └── best.pt   ← Model 3 权重
            └── 4read\
                └── weights\
                    └── best.pt   ← Model 4 权重（OBB）
```

> 如果尚无权重文件，需要自行采集数据标注并训练，详见[第八章：模型训练完整流程](#八模型训练完整流程开发者)。

---

### Step 6 准备序号标记对照表

`序号标记对照表.xlsx` 用于将 Model 4 从标签图上识别出的**数字序号**（1\~52）映射为**有意义的表计名称**（如"1号蒸汽压力表"）。

文件格式要求：

| 序号 | 表计 |
|------|------|
| 111 | 1号蒸汽压力表 |
| 113 | 2号蒸汽压力表 |
| 119 | 冷却水入口压力表 |
| ... | ... |

- 第一列列名**必须**为 `序号`，第二列列名**必须**为 `表计`
- 序号值与 Model 4 识别出的标签类别编号（YOLO 类别 ID 1\~52）对应
- 此文件需放置在两个位置：
  1. `config.py` 中 `FILE_EXCEL_PATH` 指定的路径（供 Web 端使用）
  2. `reading_batch.py` 脚本同级目录（供批量脚本使用）

---

### Step 7 启动 MySQL 并初始化数据库

**首先确认 MySQL 服务已启动：**

```bash
# Windows（XAMPP）：
#   打开 XAMPP Control Panel → MySQL 行 → 点击 [Start] 按钮

# Windows（MySQL 独立安装，命令行管理员模式）：
net start MySQL80

# Windows（图形化）：
#   Win+R → 输入 services.msc → 找到"MySQL80"→ 右键→ 启动

# Linux：
sudo systemctl start mysql
# 或
sudo service mysql start

# macOS（Homebrew）：
brew services start mysql
```

**运行数据库诊断与初始化脚本：**

```bash
cd D:\bishe
python check_db.py
```

脚本分四步执行，每步都有详细输出：

```
Step 1 / 4 · 检测 MySQL 端口是否可达
  ✔ localhost:3306 端口可达，MySQL 服务正在运行

Step 2 / 4 · 验证用户名和密码
  ✔ 用户 'root' 登录成功

Step 3 / 4 · 创建/确认数据库 BISHE
  ✔ 数据库 BISHE 已就绪

Step 4 / 4 · 创建数据表 & 默认账号
  ✔ user 表已就绪
  ✔ yolo 表已就绪
  ✔ 所有数据表初始化完成

✅  全部检测通过！
```

任何步骤出错，脚本会打印具体错误码和针对不同平台的详细修复建议，按提示操作后重新运行即可。常见错误处理详见[第十三章：常见问题](#十三常见问题与排错指南)。

---

### Step 8 启动 Web 服务

```bash
cd D:\bishe
python app.py
```

程序启动时会自动：
1. 从 `config.py` 加载所有配置
2. 读取 `序号标记对照表.xlsx` 到内存字典
3. 初始化数据库（幂等操作，重复运行不会重复建表或覆盖已有数据）
4. 创建默认管理员账号 `admin / admin123`（仅在不存在时创建）
5. 检测是否安装了 `pyOpenSSL`，自动决定以 HTTPS 或 HTTP 模式启动

**成功启动的终端输出示例：**

```
[CONFIG] ✔ 已从 config.py 加载配置
[DICT] ✔ 序号对照表加载 30 条
[DB] 正在初始化数据库...
[DB] ✔ 数据库初始化完成
============================================================
[SSL] ✔ 检测到 pyOpenSSL，已启用 HTTPS 自签名证书
[SSL]   本机访问：  https://127.0.0.1:5000
[SSL]   局域网访问：https://192.168.x.x:5000
[SSL]   ⚠ 首次访问时浏览器会提示证书不受信任，
[SSL]     点击「高级」→「继续访问（不安全）」即可正常使用摄像头。
============================================================
 * Running on https://0.0.0.0:5000
 * Debug mode: on
```

若未安装 `pyOpenSSL`，则以 HTTP 模式启动，本机访问地址为 `http://127.0.0.1:5000`（摄像头功能仅本机可用）。

> 请保持终端窗口打开，关闭终端即停止 Web 服务。

---

### Step 9 登录与首次使用

打开浏览器，访问以下地址：

```
HTTP 模式：http://localhost:5000/login
HTTPS 模式：https://localhost:5000/login
```

使用系统默认账号登录：

| 用户名 | 密码 | 权限级别 |
|--------|------|---------|
| `admin` | `admin123` | super_admin（超级管理员） |

> ⚠ **安全提示**：正式投入使用前，请通过数据库管理工具将默认密码修改为强密码，并根据实际需求创建不同权限的用户账号。

---

## 七、各功能模块详细使用说明

### 7.1 Web 端（Flask）

#### 7.1.1 单张图片检测

1. 登录后自动跳转至主页 `/index`
2. 在上传区域，通过以下任一方式提供图片：
   - 点击"选择文件"按钮，从本地文件系统选择图片（支持 jpg / jpeg / png / bmp）
   - 将图片文件直接拖拽到上传区域
   - 点击"拍照"按钮，调用摄像头实时拍摄（需 HTTPS 或 localhost 环境）
3. 点击"开始检测"，页面日志区会实时滚动显示五个步骤的进展，系统后台依次执行：

   | 步骤 | 使用模型 | 操作说明 |
   |------|---------|---------|
   | 步骤 1 | Model 1 | 从原图检测并裁剪仪表盘区域（含编号标签） |
   | 步骤 2 | Model 2 | 从步骤 1 结果中裁剪纯表盘（无标签） |
   | 步骤 3 | Model 3 | 从步骤 1 结果中裁剪编号标签区域 |
   | 步骤 4 | Model 4 | 对步骤 3 标签图进行 OBB 检测，识别仪表序号 |
   | 步骤 5 | Model 4 | 对步骤 2 表盘图进行 OBB 检测，计算刻度角度，输出读数 |

4. 检测完成后，页面展示：
   - **仪表盘带标签图**（obb_img）：Model 1 输出，确认定位是否正确
   - **纯表盘图**（dial_img）：Model 2 输出，确认表盘裁剪是否完整
   - **拟合结果图**（fitting_img）：在表盘图上标注 6 个关键几何点（详见[第九章](#九核心读数算法详解)）
   - **表计名称**（serial_number）：查序号对照表得到
   - **修正前读数**（reading_before）：仅用指针/起止刻度计算，作为参考
   - **修正后读数**（reading_after）：利用中间参考刻度修正圆心偏差，**这是最终有效读数**

5. 操作按钮：
   - **确认**：读数正确，点击后数据库记录 `is_confirmed` 更新为 1，并在 `Result_pointer.txt` 末尾追加 `confirmed.` 记录
   - **修改**：读数有误，在输入框填写正确数值后提交，数据库中 `reading_after` 更新为手动修正值，并在日志中记录 `corrected:<值>`
   - **清除日志**：清空 `Result_pointer.txt` 文件内容（不影响数据库记录）

#### 7.1.2 历史记录查询

在主页点击"历史记录"标签，展示当前登录用户的所有检测记录，含字段：

- 任务 ID、表计名称、修正前读数、最终读数、检测状态（pending/running/success/failed）、是否已确认、检测完成时间、记录创建时间

支持分页浏览（API 参数：`page` 页码，`size` 每页条数，最大 50）。

#### 7.1.3 单表计历史趋势图

在历史记录页面输入表计名称（`serial_number`），点击"查询趋势"，系统从数据库中拉取该表计所有 `detect_status='success'` 的历史记录，在页面内渲染折线趋势图，横轴为检测时间，纵轴为最终读数。

---

### 7.2 PyQt5 桌面端（单张读数）

桌面端适合无网络环境、不需要数据库存储或需要快速现场读数的场景。

**启动前配置：**

打开 `pyqt_pressure_HW_yolo_obb.py`，找到 `__init__` 方法，修改以下路径变量使其与实际路径一致：

```python
# 第 25 行：Result_pointer.txt 的保存目录（注意最后有斜杠）
self.flie_path = 'D:/毕设/ultralytics/z_pressure_HW_yolo_obb/'

# 第 26 行：序号标记对照表路径（与脚本同级时填文件名即可）
self.file_excel_path = '序号标记对照表.xlsx'

# 第 35~38 行：4 个模型权重路径（相对或绝对路径均可）
self.model_path1 = 'weight/1biaopan_all/weights/best.pt'
self.model_path2 = 'weight/2biaopan_nolabel/weights/best.pt'
self.model_path3 = 'weight/3biaopan_label/weights/best.pt'
self.model_path4 = 'weight/4read/weights/best.pt'
```

**启动命令：**

```bash
python pyqt_pressure_HW_yolo_obb.py
```

**界面布局（1800×1250 窗口，三栏结构）：**

| 左栏（1/6 宽） | 中栏（2/6 宽） | 右栏（3/6 宽） |
|---------------|---------------|---------------|
| 原始图 | 带标签仪表图 | "检测结果"标题 |
| 纯表盘图 | 拟合关键点图 | 检测时间文字 |
| "导入图片"按钮 | "确认"按钮 | 修正前读数文字 |
| "开始检测"按钮 | "修改"按钮 | 修正后读数文字 |
| | | 历史日志滚动框（1 秒刷新） |
| | | "清除"按钮 |

**按钮功能说明：**

- **导入图片**：弹出系统文件选择对话框，选中图片后自动在左栏预览原始图
- **开始检测**：依次调用 4 个模型完成完整检测流程，完成后四个图片区域全部刷新，右栏显示读数结果，同时将结果写入 `Result_pointer.txt`
- **确认**：在 `Result_pointer.txt` 末尾追加一行 `The reading is correct.`
- **修改**：弹出小型输入框，用户填入正确读数后追加 `The corrected reading is: <输入值>` 到日志
- **清除**：清空 `Result_pointer.txt` 全部内容

**检测结果保存路径：**

```
outputs-YYYY-MM-DD/              ← 脚本同级目录，每天一个文件夹
└── <图片名（不含扩展名）>/
    ├── <图片名>_all.jpg           仪表区域（含标签）裁剪图
    ├── <图片名>_biaopan.jpg       纯表盘裁剪图
    ├── <图片名>_biaoqian.jpg      标签区域裁剪图
    └── <图片名>_fitting.jpg       拟合关键点标注图
```

---

### 7.3 批量图片读数脚本

适用于已有一批现场照片需要一次性批量处理的场景，无需人工逐张操作。

**使用前检查：**

确认 `reading_batch.py` 中以下路径变量正确：

```python
file_path_all      = 'data_show'           # 待读数图片输入目录
file_excel_path    = '序号标记对照表.xlsx'  # 序号对照表路径
Reading_result_path = 'a_reading_batch_results/'  # 汇总 Excel 永久保存目录
model_path1 = 'weight/1biaopan_all/weights/best.pt'
model_path2 = 'weight/2biaopan_nolabel/weights/best.pt'
model_path3 = 'weight/3biaopan_label/weights/best.pt'
model_path4 = 'weight/4read/weights/best.pt'
```

**操作步骤：**

1. 在脚本同级目录下创建 `data_show/` 文件夹（若不存在）
2. 将所有待读数图片（.jpg / .jpeg / .png / .bmp 均可）放入 `data_show/`
3. 确认 `序号标记对照表.xlsx` 在脚本同级目录，且内容与实际仪表对应
4. 确认 `weight/` 目录下 4 个权重文件均已就位

```bash
python reading_batch.py
```

**输出文件说明：**

```
outputs/
└── outputs-YYYY-MM-DD HH-MM-SS/    ← 本次批量检测的所有中间结果
    ├── all/                          Model 1 输出：仪表区域裁剪图（所有图片）
    ├── biaopan/                      Model 2 输出：纯表盘图（按图片名分子目录）
    │   ├── img_001/
    │   └── img_002/
    ├── biaoqian/                     Model 3 输出：标签图（按图片名分子目录）
    │   ├── img_001/
    │   └── img_002/
    ├── fitcenter/                    拟合圆心标注图（所有图片）
    ├── biaoqian.txt                  中间文件：图片名 + 表计名（每行一条）
    ├── dushu.txt                     中间文件：图片名 + 读数 + 检测日期
    └── Reading results-YYYY-MM-DD.xlsx   本次汇总表格

a_reading_batch_results/
└── Reading results-YYYY-MM-DD.xlsx      同一份汇总表格的永久保存副本
```

**汇总 Excel 表格列说明：**

| 图片名称 | 表计 | 最终读数 | 检测时间 |
|---------|------|---------|---------|
| a(31) | 1号蒸汽压力表 | 0.532362 | 2025-07-20 |
| a(104) | 冷却水压力表 | 0.529226 | 2025-07-20 |
| bad_img | 读数失败 | 读数失败 | 未知 |

- 若某张图片在任意一步检测失败（模型未检出目标），对应行"最终读数"显示"读数失败"，"检测时间"显示"未知"，**不影响其他图片的处理**
- 汇总表格已做居中对齐和列宽优化，可直接用于汇报提交

---

### 7.4 多日期趋势图生成

适用于已积累多日批量读数 Excel 文件，想直观查看各表计读数随时间变化的趋势。

**pic_all.py — 各表计独立趋势图（每表计单独一张图）**

```bash
# 步骤 1：将各日期生成的汇总 Excel 文件放入
#   a_reading_results_diff_day/
# 文件名格式无要求，能被 glob 匹配 *.xlsx 即可

# 步骤 2：运行脚本
python pic_all.py

# 步骤 3：查看生成的趋势图
#   a_pic_results_all/<表计名>的读数变化趋势图.png
```

生成的图表特点：每个表计一张独立折线图，尺寸 20×12 英寸，每个数据点旁标注具体读数值，横轴以日期为刻度（每天一格）。

**pic_one.py — 所有表计合并趋势图（所有表计在同一张图）**

```bash
# 步骤 1：将各日期的汇总 Excel 文件放入
#   a_reading_results_show_pic_one/

# 步骤 2：运行脚本
python pic_one.py

# 步骤 3：查看结果
#   a_pic_results_one/读数变化趋势图YYYY-MM-DD.png
```

生成的图表特点：所有表计的折线绘制在同一张图中，用不同颜色区分，适合快速对比各表计读数趋势。

**输入 Excel 格式要求（供两个脚本读取）：**

- 需包含列：`图片名称`、`表计`、`最终读数`、`检测时间`
- `检测时间` 列需能被 `pd.to_datetime()` 解析（推荐格式：`2025-07-20`）
- `最终读数` 列为数值，值为"读数失败"的行会被自动过滤

---

## 八、模型训练完整流程（开发者）

如果需要在自己采集的数据集上重新训练模型，请按以下流程操作。

### 8.1 数据标注说明

**推荐标注工具：** [RoLabelImg](https://github.com/cgvict/roLabelImg)

支持旋转框（robndbox，对应 OBB 检测）和普通矩形框（bndbox），保存为 XML 格式。

**各模型标注类别规范：**

| 用途 | 类别名 | 说明 |
|------|--------|------|
| Model 1 | `Instrument` | 整个仪表区域（含编号标签），矩形框标注 |
| Model 2 | `Pointer` | 纯表盘区域（圆形表盘，不含标签），矩形框标注 |
| Model 3 | `Label` | 仪表编号标签区域，矩形框标注 |
| Model 4 | `Scale` | 起始（零）刻度线，**旋转框**标注 |
| Model 4 | `Scale2` | 终止（满量程）刻度线，**旋转框**标注 |
| Model 4 | `Pointer` | 指针，**旋转框**标注 |
| Model 4 | `1`～`52` | 标签上的序号数字（对应 YOLO 类别 ID 1\~52），**旋转框**标注 |

> Model 4 的 `Scale`、`Scale2`、`Pointer` 和中间参考刻度（类别 ID 最小值，对应`Scale`排序后第三项）的旋转框必须能清晰区分刻度线的延伸方向，这是角度计算精度的关键。

### 8.2 标注数据转换

**第一步：RoLabelImg XML → DOTA XML → DOTA TXT**

修改 `z_1_xml2data.py` 中路径变量后运行：

```python
# z_1_xml2data.py 中修改以下三个路径：
roxml_path   = r'datasets/pressure_4read_xml2txt/labels/xml'    # 原始 XML 标注文件目录
dotaxml_path = r'datasets/pressure_4read_xml2txt/labels/dota'   # 中间 DOTA XML 输出目录
out_path     = r'datasets/pressure_4read_xml2txt/labels/txt'    # DOTA TXT 输出目录

# cls_list 修改为实际使用的类别列表，顺序需与标注时一致
cls_list = ['Scale', '1', '2', ..., '52', 'Scale2', 'Pointer']
```

```bash
python z_1_xml2data.py
```

脚本同时处理旋转框（robndbox，包含 `cx, cy, w, h, angle`）和矩形框（bndbox，包含 `xmin, ymin, xmax, ymax`），统一转换为四点坐标格式 `(x0,y0), (x1,y1), (x2,y2), (x3,y3)`。

**第二步：DOTA TXT → YOLO-OBB 归一化 TXT**

将第一步输出的 TXT 文件分别复制到对应的 `train_original/` 和 `val_original/` 目录后运行：

```bash
python z_2_txt2txt.py
```

脚本读取对应图片，将绝对像素坐标归一化到 [0, 1]，输出格式为 YOLO-OBB 标准格式：

```
<class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> <x3_norm> <y3_norm> <x4_norm> <y4_norm>
```

输出至 `datasets/pressure_4read_xml2txt/labels/train/` 和 `.../val/`。

### 8.3 训练 Model 1 / 2 / 3

为每个模型准备 YAML 数据集配置文件（参考 `yaml/` 目录下已有文件，格式遵循 Ultralytics 规范），然后修改 `dataset_train.py` 中对应行的注释，运行训练：

```python
# dataset_train.py 中取消注释对应行（每次只训练一个模型）：
model.train(data='yaml/dataset_1biaopan_all.yaml', workers=0, epochs=200, batch=8)
# model.train(data='yaml/dataset_2biaopan_nolabel.yaml', workers=0, epochs=200, batch=8)
# model.train(data='yaml/dataset_3biaopan_label.yaml', workers=0, epochs=200, batch=8)
```

```bash
python dataset_train.py
```

训练结果默认保存至 `runs/detect/train*/weights/best.pt`，训练过程中可在 `runs/detect/train*/` 下查看损失曲线和 mAP 等指标。

**推荐训练超参数（参考）：**

| 参数 | 值 | 说明 |
|------|---|------|
| `epochs` | 200 | 训练轮次 |
| `batch` | 8 | 批量大小（显存不足时降低至 4） |
| `workers` | 0 | Windows 下必须设为 0，否则 DataLoader 多进程报错 |
| `conf` | 0.7 | 推理置信度阈值 |

### 8.4 训练 Model 4（OBB 读数模型）

Model 4 使用 YOLOv11-OBB 架构，训练命令封装在 `z_3train_read.py`：

```python
# z_3train_read.py 内容（可直接修改）：
model = YOLO('yolo11-obb.yaml').load('yolo11n-obb.pt')  # 从预训练权重初始化
model.train(data='yaml/dataset_4read.yaml', epochs=200, batch=8, workers=0)
```

```bash
python z_3train_read.py
```

训练前请确认：
- `yaml/dataset_4read.yaml` 中 `nc`（类别数）和 `names` 列表与 `z_1_xml2data.py` 中的 `cls_list` 完全一致
- `yolo11n-obb.pt` 预训练权重已下载（首次运行时 Ultralytics 会自动从网络下载）

### 8.5 测试验证

```bash
python z_4_test_read.py
```

对 `datasets/pressure_4read/images/train/` 中的图片进行推理，保存检测结果至 `runs/detect/predict*/`。可视化检查旋转框是否准确覆盖刻度线和指针，判断模型质量。

---

## 九、核心读数算法详解

Model 4 输出每个检测目标的**旋转边界框（OBB）**，包含四个角点的归一化坐标（9 列数据：类别 ID + 8 个坐标值）。读数算法基于几何方法，利用这些 OBB 坐标计算指针的角度比例，转换为物理读数。

### 9.1 OBB 坐标预处理

Model 4 每行输出格式：

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

各目标的坐标先按类别 ID 升序排序，然后反归一化（乘以图像宽高）得到像素坐标。四个角点的**均值中心**作为各目标的代表点：

```python
xsf = (x_s1 + x_s2 + x_s3 + x_s4) / 4  # 起始刻度中心
ysf = (y_s1 + y_s2 + y_s3 + y_s4) / 4
```

### 9.2 圆心估算（双方法融合，提高精度）

**OBB 短边中点延长线法：** 将 OBB 四条边按长度升序排列，取两条**最短边**各自的中点，连接两个中点形成穿过 OBB 轴线的直线。

刻度线和指针是长条形目标，其 OBB 的长轴方向即指向圆心，因此各目标的轴线（短边中点连线延长后）会汇聚在表盘圆心附近。

**圆心估算 1（`cxx, cyy`）**：使用 Scale（起始刻度）、Scale2（终止刻度）、中间参考刻度三者的轴线两两求交，取三个交点中距离最近的两点均值作为圆心。此估算利用了三条刻度线，不受指针位置影响。

**圆心估算 2（`cx, cy`）**：使用 Scale、Scale2、Pointer（指针）三者的轴线两两求交，同样取最近两点均值。此估算直接利用指针位置，对读数结果更敏感。

**最终圆心**（取两组均值）：

```
acx = (cx + cxx) / 2
acy = (cy + cyy) / 2
```

### 9.3 角度比例法计算读数

以最终圆心为原点，利用向量**顺时针夹角**公式（叉积判断方向，点积计算角度）：

```
θ₁ = 顺时针角度(起始刻度方向向量, 指针方向向量)
θ₂ = 顺时针角度(起始刻度方向向量, 终止刻度方向向量)
θ₃ = 顺时针角度(起始刻度方向向量, 中间参考刻度方向向量)

修正前读数 = θ₁ / θ₂ × 量程范围 + 量程起点
           = θ₁ / θ₂ × 1 - 0.1   （量程 -0.1 ~ 0.9 MPa）

修正后读数 = θ₁'/θ₂' × 1 - 0.1   （使用最终圆心 acx, acy 重算）

最终读数（二次修正）= 修正后读数 + (0.4 - readValue3) / 2
```

其中 `readValue3 = θ₃/θ₂ × 1 - 0.1` 是以当前圆心计算得到的中间参考刻度处理论读数。由于中间参考刻度的真实物理值已知（当前设定为 0.4 MPa），`(0.4 - readValue3)` 即圆心偏差引起的系统误差，除以 2 后补偿到最终读数，进一步提高精度。

### 9.4 拟合结果图标注说明

`_fitting.jpg` 图像上标注了 6 个关键几何点：

| 颜色（BGR） | 含义 |
|------------|------|
| 红色 `(0,0,255)` | Scale 与 Scale2 轴线交点（圆心候选） |
| 深红 `(0,0,120)` | Scale 与 Pointer 轴线交点（圆心候选） |
| 黑色 `(0,0,0)` | Scale2 与 Pointer 轴线交点（圆心候选） |
| 红色 `(0,0,255)` | 圆心估算 2（cx, cy） |
| 红色 `(0,0,255)` | 中间参考刻度（0.4 MPa）位置 |
| 红色 `(0,0,255)` | 圆心估算 1（cxx, cyy） |

---

## 十、API 接口说明

以下为 Flask 提供的所有主要接口，均需已登录（Session 有效）才能访问（标注"公开"的除外）：

| 方法 | 路径 | 权限 | 说明 |
|------|------|------|------|
| GET | `/login` | 公开 | 返回登录页面 HTML |
| POST | `/login` | 公开 | 验证账号密码；成功后 Session 写入 user_id / username，重定向至 `/index` |
| GET | `/logout` | 需登录 | 清除 Session，重定向至 `/login` |
| GET | `/index` | 需登录 | 返回主页 HTML |
| POST | `/upload` | 需登录 | 接收图片文件，创建检测任务，启动后台检测线程，返回 `{"task_id": "YYYY-MM-DD-N"}` |
| GET | `/poll/<task_id>` | 需登录 | 轮询任务进度，返回 status / logs / img_obb / img_dial / img_fitting / reading_before / reading_after / serial_number / error |
| POST | `/confirm` | 需登录 | 确认读数，body: `{"task_id": "..."}` |
| POST | `/modify` | 需登录 | 手动修正读数，body: `{"task_id": "...", "value": "0.532"}` |
| POST | `/clear` | 需登录 | 清空 TXT 日志文件 |
| GET | `/get_log` | 需登录 | 返回 TXT 日志文件全文内容，`{"content": "..."}` |
| GET | `/api/history` | 需登录 | 分页历史记录，参数：`page`（页码，默认 1）、`size`（每页条数，默认 20，最大 50） |
| GET | `/api/serial_history` | 需登录 | 指定表计的历史趋势数据，参数：`serial`（表计名称）、`limit`（最多条数，默认 60，最大 200） |
| GET | `/image/<path>` | 需登录 | 提供检测结果图片文件访问（安全限制：仅允许访问 uploads/ 和 outputs/ 目录内的文件） |

**接口返回格式（JSON）示例：**

`/poll/<task_id>` 返回：

```json
{
  "status": "success",
  "logs": ["▶ 步骤 1/5...", "✔ 步骤 1 完成", "..."],
  "img_obb": "/image/outputs/outputs-2026-04-05/2026-04-05-1/img_all.jpg",
  "img_dial": "/image/outputs/outputs-2026-04-05/2026-04-05-1/img_biaopan.jpg",
  "img_fitting": "/image/outputs/outputs-2026-04-05/2026-04-05-1/img_fitting.jpg",
  "detect_time": "2026-04-05 16:43:20",
  "serial_number": "1号蒸汽压力表",
  "reading_before": 0.537795,
  "reading_after": 0.535530,
  "error": null
}
```

---

## 十一、用户权限体系

系统预设三级权限，存储于 `user.user_level` 字段：

| 权限级别 | 标识 | 说明 |
|---------|------|------|
| 超级管理员 | `super_admin` | 系统最高权限，初始 `admin` 账号即为此级别 |
| 管理员 | `admin` | 中级管理权限 |
| 普通用户 | `user` | 基础使用权限，新注册用户默认此级别 |

当前版本所有已登录用户均可访问全部功能（上传检测、历史查询、确认/修改读数）。三级权限结构已预留扩展空间，如需对不同级别用户限制特定接口（例如仅 admin 可删除历史记录、仅 super_admin 可管理用户），可在各路由的 `login_required` 装饰器处增加判断逻辑。

---

## 十二、输出文件说明

### 12.1 Result_pointer.txt（读数历史日志）

每次检测成功后，系统自动向 `TXT_LOG_PATH`（由 `config.py` 配置）追加写入记录。

**Web 端写入格式（UTF-8 编码）：**

```
2026-04-05 16:43:17
任务编号：2026-04-05-1  操作人：admin
a(6) 1号蒸汽压力表
修正前读数为0.537795
修正后读数为0.535530

[admin][2026-04-05-1] confirmed. 2026-04-05 16:45:00
[admin][2026-04-05-2] corrected:0.53 2026-04-05 17:00:00
```

**PyQt5 桌面端写入格式（GBK 编码）：**

```
2025-07-20 19-25-26
a(31) 1号蒸汽压力表
修正前读数为0.533844
修正后读数为0.532362
The reading is correct.
```

> 注意：Web 端与桌面端使用不同编码（UTF-8 vs GBK），混合使用同一日志文件时可能出现乱码，建议按需分开使用或统一编码。

### 12.2 outputs/ 目录结构

```
outputs/
└── outputs-YYYY-MM-DD/           按日期分组
    └── <任务ID>/                 每次 Web 端检测的专属目录
        ├── <图片名>_all.jpg       Model 1 输出：仪表区域（含标签）裁剪图
        ├── <图片名>_biaopan.jpg   Model 2 输出：纯表盘裁剪图
        ├── <图片名>_biaoqian.jpg  Model 3 输出：标签区域裁剪图
        ├── <图片名>_fitting.jpg   关键点拟合标注图（含 6 个彩色点）
        └── result.txt             单任务读数记录文本
```

---

## 十三、常见问题与排错指南

### ❌ MySQL 端口不可达（错误 2003）

**现象：** `check_db.py` 输出 `无法连接到 localhost:3306`，或启动 `app.py` 后上传图片返回"数据库连接失败"。

**解决：** MySQL 服务未启动。

```bash
# Windows（XAMPP）：
#   打开 XAMPP Control Panel → MySQL 行 → 点击 [Start]

# Windows（独立 MySQL，管理员命令行）：
net start MySQL80   # 服务名可能是 MySQL 或 MySQL80，以实际为准

# Linux：
sudo systemctl start mysql

# macOS（Homebrew）：
brew services start mysql
```

启动后，重新运行 `python check_db.py` 验证。

---

### ❌ 密码错误（错误 1045）

**现象：** `check_db.py` 输出 `登录失败（错误 1045）：Access denied for user 'root'`

**解决：** 打开 `config.py`，将 `DB_PASSWORD` 修改为正确的 MySQL 密码。

| 安装方式 | 默认密码 |
|---------|---------|
| XAMPP | `""` （空字符串） |
| MySQL Installer / Workbench | 安装时自设的密码（无默认值） |
| 宝塔面板 | 宝塔数据库密码（在宝塔面板数据库页面可查） |

---

### ❌ 序号标记对照表加载失败

**现象：** 终端输出 `[DICT] ⚠ 序号标记对照表加载失败`

**检查：**
1. `config.py` 中 `FILE_EXCEL_PATH` 路径是否正确（可改为绝对路径排查）
2. Excel 文件是否损坏（尝试用 WPS / Excel 直接打开）
3. 列名是否准确：第一列必须是 `序号`，第二列必须是 `表计`
4. Excel 文件编码是否正常（另存为标准 xlsx 格式）

---

### ❌ 检测结果页面无读数 / 读数为 NaN

**现象：** Web 端状态变为 success 但读数显示空或 NaN；桌面端检测后读数显示异常值

**排查步骤：**

1. 查看 Web 端 `/poll/<task_id>` 返回的 `error` 字段，或查看终端日志中的报错信息
2. 检查图片质量：是否清晰、表盘是否正对镜头、光照是否均匀无强烈反光
3. 确认 `config.py` 中 4 个 `MODEL_PATH*` 路径均指向实际存在的 `.pt` 文件
4. 尝试降低置信度阈值：在 `_run_detection` 函数（app.py）中将各步骤 `conf=0.7` 改为 `conf=0.5`，重启服务后重试
5. 确认 Model 4 能同时检测到 `Scale`（类别排序最小）、`Scale2`、`Pointer` 三类；任何一类缺失均会导致计算失败

---

### ❌ 摄像头无法调用

**现象：** 浏览器点击"拍照"按钮无反应，或控制台报 `NotAllowedError`

**原因：** 浏览器的 `getUserMedia` 摄像头 API 仅在**安全上下文**（HTTPS 或 `localhost`）下可用。通过局域网 IP（如 `http://192.168.x.x:5000`）以 HTTP 访问时，摄像头不可用。

**解决：**

```bash
pip install pyOpenSSL
# 重启 app.py，以 HTTPS 模式启动

# 局域网设备访问：https://192.168.x.x:5000
# 首次访问浏览器提示"不安全"→ 点击"高级"→ "继续访问"
```

---

### ❌ 结果图片无法显示（404）

**现象：** 检测完成后，页面图片区域显示"图片加载失败"，浏览器 Network 面板中图片请求返回 404

**可能原因：**
- Windows 下路径分隔符为反斜杠 `\`，系统会自动转换为正斜杠 `/`，若仍有问题，检查 `OUTPUT_FOLDER` 路径中是否含有特殊字符
- `outputs/` 目录是否有正确的读写权限（Linux / macOS 下执行 `chmod 755 outputs`）
- 检查终端日志中是否有 `[SERVE] ⚠ 路径不可访问` 的输出，并对照日志中的 `abs_path` 和 `allowed` 进行排查

---

### ❌ PyQt5 桌面端启动后立即崩溃

**现象：** 运行 `pyqt_pressure_HW_yolo_obb.py` 后窗口一闪而过或终端直接退出

**解决：**
1. 确认已安装 PyQt5：`pip install PyQt5`
2. 检查 `__init__` 方法中 4 个模型路径是否正确（文件不存在时 YOLO 初始化会抛出异常）
3. 检查 `序号标记对照表.xlsx` 是否存在且格式正确（第一行必须是标题行，含 `序号` 和 `表计` 两列）
4. 如有报错，在终端中执行 `python pyqt_pressure_HW_yolo_obb.py` 查看具体错误信息

---

### ❌ 批量读数大量图片"读数失败"

**现象：** 批量运行 `reading_batch.py` 后，Excel 结果中多行显示"读数失败"

**说明：** 正常现象，不影响其他成功图片。"读数失败"通常是某张图片在某步骤中模型未检测到目标（置信度低于阈值）。

**建议：**
- 对失败图片用 Web 端单独上传重试（单张可手动观察中间结果图）
- 将 `reading_batch.py` 中各 `yolo()` 调用的 `conf=0.7` 降低至 `conf=0.5`
- 检查失败图片的质量：表盘是否清晰可见、是否存在严重遮挡或过曝

---

## 十四、注意事项与已知限制

**量程硬编码：** 当前读数算法中量程固定为 -0.1\~0.9 MPa（满量程 1 MPa）。公式体现在各脚本的 `× 1 - 0.1` 处。若现场压力表量程不同（如 0\~1.6 MPa），需同时修改 `app.py`、`reading_batch.py`、`pyqt_pressure_HW_yolo_obb.py` 三处的读数计算公式，以及中间参考刻度的真实值（当前设定为 0.4 MPa）。

**仪表编号上限：** Model 4 支持的标签序号类别为 `1`\~`52`，即最多支持 52 个不同仪表编号。若现场仪表数量超过 52 台，需重新标注并训练 Model 4。

**并发检测性能：** Web 端每次检测在独立线程中运行（`threading.Thread`），支持多用户同时提交任务，但 GPU 显存有限。多任务并发时 YOLO 模型会排队执行，总耗时会成倍增加。生产环境建议限制同时进行的检测任务数量，或使用任务队列（如 Celery + Redis）管理。

**模型加载方式：** 每次调用 `YOLO(model_path)` 都会重新加载模型到内存，5 个步骤合计约增加 5\~15 秒冷启动时间。可在 `app.py` 启动时预加载所有模型为全局变量（`YOLO_M1 = YOLO(MODEL_PATH1)` 等）来提升推理速度，但会增加启动时的内存占用。

**日志编码混用：** Web 端写入 `Result_pointer.txt` 使用 UTF-8，桌面端使用 GBK。混合使用同一文件时请注意编码一致性，推荐为两套工具分别配置不同的 `TXT_LOG_PATH`。

**HTTPS 证书信任：** `pyOpenSSL` 生成的是自签名证书，浏览器会显示"不安全"警告。点击"高级"→"继续访问"后即可正常使用，这是自签名证书的预期行为，不影响数据传输安全性（仅 localhost 环境使用）。正式生产部署如需消除警告，可申请正式 SSL 证书（如 Let's Encrypt）并在 Flask 或 Nginx 中配置。

**Windows 反斜杠兼容：** `app.py` 中所有路径在存入数据库前均会通过 `_fwdpath()` 函数将反斜杠转为正斜杠，访问图片时也做了双路径候选处理，基本兼容 Windows 路径。若遇到图片 404 问题，优先检查路径中是否含有中文或空格。

**生产环境建议：** 正式部署时请将 `config.py` 中 `FLASK_DEBUG` 改为 `False`，使用 Gunicorn（Linux）或 Waitress（Windows）替代 Flask 内建开发服务器，并通过 Nginx 做反向代理，以提升稳定性和并发处理能力。

---

> **遇到任何数据库相关问题，请优先运行：**
>
> ```bash
> python check_db.py
> ```
>
> 脚本会自动定位问题并给出对应平台的详细修复建议。
