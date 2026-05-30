# 图标资源说明

本目录用于存放小程序所需的图标资源。

## 必需的图标文件

### TabBar图标（81x81px，推荐使用@2x: 162x162px）

| 文件名 | 用途 | 建议颜色 |
|--------|------|---------|
| camera.png | 检测Tab（未选中） | #7A7E83 |
| camera-active.png | 检测Tab（选中） | #3cc51f |
| history.png | 历史Tab（未选中） | #7A7E83 |
| history-active.png | 历史Tab（选中） | #3cc51f |
| profile.png | 我的Tab（未选中） | #7A7E83 |
| profile-active.png | 我的Tab（选中） | #3cc51f |

### 页面图标（建议48x48px或更大）

| 文件名 | 用途 |
|--------|------|
| logo.png | 登录页logo（160x160rpx） |
| user.png | 用户图标 |
| lock.png | 密码图标 |
| eye-open.png | 显示密码 |
| eye-close.png | 隐藏密码 |
| serial.png | 序列号图标 |
| upload.png | 上传图标 |
| camera-btn.png | 拍照按钮图标 |
| album.png | 相册图标 |
| close.png | 关闭图标 |
| result.png | 结果图标 |
| task.png | 任务图标 |
| check.png | 确认图标 |
| check-small.png | 小确认图标 |
| empty.png | 空状态图（200x200rpx） |
| avatar.png | 默认头像（120x120rpx） |
| arrow-right.png | 右箭头 |
| history-icon.png | 历史图标 |
| log-icon.png | 日志图标 |
| download-icon.png | 下载图标 |
| server-icon.png | 服务器图标 |
| version-icon.png | 版本图标 |

## 快速获取图标

### 方案1：在线图标库（推荐）

1. **iconfont阿里图标库**
   - 网址：https://www.iconfont.cn/
   - 搜索关键词下载PNG图标
   - 可调整颜色和大小

2. **icons8**
   - 网址：https://icons8.com/
   - 提供多种风格图标
   - 支持自定义颜色

3. **flaticon**
   - 网址：https://www.flaticon.com/
   - 海量免费图标

### 方案2：使用图标字体

暂不推荐，小程序对字体支持有限

### 方案3：临时占位（开发阶段）

可以暂时使用纯色方块或文字替代：

```
<!-- 在wxml中临时替换 -->
<!-- 原：<image src="/images/user.png"></image> -->
<!-- 改：<view class="icon-placeholder">👤</view> -->
```

## 制作规范

- **格式**: PNG（支持透明）
- **尺寸**: 建议使用@2x或@3x适配高清屏
- **命名**: 小写字母+中划线，如 camera-active.png
- **文件大小**: 单个图标 < 50KB
- **背景**: TabBar图标建议透明背景

## 示例CSS图标（临时方案）

如果暂时没有图标，可以使用CSS绘制简单图形：

```wxss
/* 圆形占位 */
.icon-circle {
  width: 40rpx;
  height: 40rpx;
  border-radius: 50%;
  background-color: #999;
}

/* 方形占位 */
.icon-square {
  width: 40rpx;
  height: 40rpx;
  background-color: #999;
}
```

## 注意事项

1. TabBar图标必须存在，否则小程序无法运行
2. 其他页面图标可以暂时用emoji或文字替代
3. 所有图标建议统一风格
4. 注意版权问题，使用免费或已授权图标

---

准备好图标后，将所有文件放入本目录即可。
