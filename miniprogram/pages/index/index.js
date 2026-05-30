// pages/index/index.js
const app = getApp();
const { detectApi } = require('../../utils/api.js');
const { showToast, showLoading, hideLoading, showConfirm, getImageUrl } = require('../../utils/util.js');

Page({
  data: {
    serialNumber: '',
    imageUrl: '',
    imagePath: '',
    detecting: false,
    progress: 0,
    statusText: '准备检测...',
    result: null,
    statusMap: {
      'pending': '等待中',
      'running': '检测中',
      'success': '成功',
      'failed': '失败',
      'uploaded': '已上传'
    }
  },

  onLoad() {
    if (!app.isLoggedIn()) {
      wx.redirectTo({ url: '/pages/login/login' });
    }
  },

  onShow() {
    if (!app.isLoggedIn()) {
      wx.redirectTo({ url: '/pages/login/login' });
    }
  },

  // =========================
  // 输入序列号
  // =========================
  onSerialInput(e) {
    this.setData({
      serialNumber: e.detail.value
    });
  },

  // =========================
  // 拍照 / 选图
  // =========================
  takePhoto() {
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['camera'],
      success: (res) => {
        this.setData({
          imageUrl: res.tempFilePaths[0],
          imagePath: res.tempFilePaths[0],
          result: null
        });
      }
    });
  },

  chooseImage() {
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['album'],
      success: (res) => {
        this.setData({
          imageUrl: res.tempFilePaths[0],
          imagePath: res.tempFilePaths[0],
          result: null
        });
      }
    });
  },

  removeImage() {
    this.setData({
      imageUrl: '',
      imagePath: '',
      result: null
    });
  },

  // =========================
  // ✅ 开始检测（核心流程）
  // =========================
  startDetect() {
    if (!this.data.imagePath) {
      showToast('请先选择图片', 'none');
      return;
    }
  
    this.setData({
      detecting: true,
      progress: 0,
      statusText: '上传图片中...'
    });
  
    this.updateProgress(10);
  
    let taskId = null;
  
    detectApi.upload(this.data.imagePath)
      .then(res => {
        console.log("上传返回：", res);
  
        // 上传失败直接抛出
        if (res.error) {
          throw new Error(res.error);
        }
        if (!res.task_id) {
          throw new Error("上传失败：未获取到任务ID");
        }
  
        taskId = res.task_id;
        this.setData({ statusText: '启动检测中...' });
        this.updateProgress(30);
  
        return detectApi.startDetect(taskId);
      })
      .then(() => {
        this.setData({ statusText: '检测中，请稍候...' });
        this.pollResult(taskId, 0);
      })
      .catch(err => {
        console.error('检测失败:', err);
  
        // ✅ 关键：上传失败立刻停止所有状态
        this.setData({
          detecting: false,
          progress: 0,
          statusText: '上传失败'
        });
  
        // ✅ 友好提示，不暴露数据库错误
        showToast('上传失败，请稍后重试', 'none');
      });
  },

  pollResult(taskId, attempts) {
    const maxAttempts = 60;
  
    if (attempts >= maxAttempts) {
      this.setData({ detecting: false, progress: 0 });
      showToast('检测超时，请重试', 'none');
      return;
    }
  
    const progress = Math.min(30 + (attempts / maxAttempts) * 60, 90);
    this.setData({
      progress,
      statusText: `检测中... ${Math.floor(progress)}%`
    });
  
    setTimeout(() => {
      detectApi.poll(taskId)
        .then(res => {
          const result = {
            ...res,
            task_id: taskId,
            obb_img_path: res.img_obb,
            dial_img_path: res.img_dial,
            label_img_path: res.img_fitting
          };
  
          // 只要有读数，就认为成功（最稳）
          const hasReading = res.reading != null || res.reading_before != null;
  
          if (hasReading) {
            this.setData({
              result,
              detecting: false,
              progress: 100,
              statusText: '检测完成！'
            });
            showToast('检测成功', 'success');
            return;
          }
  
          // 状态判断
          const status = res.status || res.detect_status;
          if (status === 'success') {
            this.setData({
              result,
              detecting: false,
              progress: 100,
              statusText: '检测完成！'
            });
            showToast('检测成功', 'success');
          } else if (status === 'failed' || res.error) {
            this.setData({ detecting: false, progress: 0 });
            showToast(res.error || '检测失败', 'none');
          } else {
            this.pollResult(taskId, attempts + 1);
          }
        })
        .catch(() => {
          this.pollResult(taskId, attempts + 1);
        });
    }, 1500);
  },
  // =========================
  // 进度动画
  // =========================
  updateProgress(target) {
    const current = this.data.progress;
    if (current < target) {
      this.setData({ progress: current + 1 });
      setTimeout(() => this.updateProgress(target), 30);
    }
  },

  // =========================
  // 修改读数
  // =========================
  modifyReading() {
    if (!this.data.result) return;

    wx.showModal({
      title: '修改读数',
      editable: true,
      placeholderText: '请输入新的读数值',
      content: this.data.result.reading_after?.toString() || '',
      success: (res) => {
        if (res.confirm && res.content) {
          const value = parseFloat(res.content);
          if (isNaN(value)) {
            showToast('请输入有效数字', 'none');
            return;
          }

          showLoading('修改中...');
          detectApi.modify(this.data.result.task_id, value)
            .then(() => {
              hideLoading();
              showToast('修改成功', 'success');

              this.setData({
                'result.reading_after': value
              });
            })
            .catch(err => {
              hideLoading();
              showToast(err.error || '修改失败', 'none');
            });
        }
      }
    });
  },

  // =========================
  // 确认
  // =========================
  pollResult(taskId, attempts) {
    const maxAttempts = 60;
  
    // 不管怎样，先打日志看后端到底返回了啥
    console.log(`【轮询第 ${attempts} 次】taskId:`, taskId);
  
    if (attempts >= maxAttempts) {
      showToast('检测超时，请查看历史记录', 'none');
      this.setData({
        detecting: false,
        progress: 0
      });
      return;
    }
  
    const progress = Math.min(30 + (attempts / maxAttempts) * 60, 90);
    this.setData({
      progress,
      statusText: `检测中... ${Math.floor(progress)}%`
    });
  
    setTimeout(() => {
      detectApi.poll(taskId)
        .then(res => {
          // ✅ 关键：把后端返回完整打出来，你看控制台就知道结构
          console.log("【后端真实返回】", res);
  
          // 构造结果
          const result = {
            ...res,
            task_id: taskId,
            obb_img_path: res.img_obb,
            dial_img_path: res.img_dial,
            label_img_path: res.img_fitting
          };
  
          // ==============================================
          // ✅ 终极判断：只要有 reading 就视为成功
          // ==============================================
          const hasReading = (res.reading != null || res.reading_before != null);
          
          if (hasReading) {
            // 有读数 = 已经成功，直接结束
            this.setData({
              result,
              detecting: false,
              progress: 100,
              statusText: '检测完成！'
            });
            showToast('检测成功', 'success');
            return;
          }
  
          // 原有状态判断（兜底）
          const status = res.status || res.detect_status || res.state;
          if (status === 'success' || status === 'completed') {
            this.setData({
              result,
              detecting: false,
              progress: 100,
              statusText: '检测完成！'
            });
            showToast('检测成功', 'success');
          } else if (status === 'failed' || status === 'error') {
            this.setData({ detecting: false, progress: 0, statusText: '检测失败' });
            showToast(res.error || '检测失败', 'none');
          } else {
            // 继续轮询
            this.pollResult(taskId, attempts + 1);
          }
        })
        .catch(err => {
          console.error('轮询异常', err);
          this.pollResult(taskId, attempts + 1);
        });
    }, 1500);
  },

  // =========================
  // 确认结果
  // =========================
  confirmResult() {
    if (!this.data.result) return;

    showConfirm('确认当前检测结果？')
      .then(confirmed => {
        if (confirmed) {
          showLoading('确认中...');
          detectApi.confirm(this.data.result.task_id)
            .then(() => {
              hideLoading();
              showToast('已确认', 'success');

              this.setData({
                'result.is_confirmed': true
              });
            })
            .catch(err => {
              hideLoading();
              showToast(err.error || '确认失败', 'none');
            });
        }
      });
  },

  // =========================
  // 查看详情
  // =========================
  viewDetail() {
    if (!this.data.result) return;

    wx.navigateTo({
      url: `/pages/detail/detail?taskId=${this.data.result.task_id}`
    });
  },

  // =========================
  // 预览图片（修复）
  // =========================
  previewImage(e) {
    const url = e.currentTarget.dataset.url;

    const urls = [];

    if (this.data.result.obb_img_path) {
      urls.push(getImageUrl(this.data.result.obb_img_path));
    }
    if (this.data.result.dial_img_path) {
      urls.push(getImageUrl(this.data.result.dial_img_path));
    }
    if (this.data.result.label_img_path) {
      urls.push(getImageUrl(this.data.result.label_img_path));
    }

    if (urls.length === 0) {
      showToast('暂无图片', 'none');
      return;
    }

    wx.previewImage({
      current: url,
      urls
    });
  },

  getImageUrl(path) {
    return getImageUrl(path);
  }
});