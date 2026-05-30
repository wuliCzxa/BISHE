// pages/profile/profile.js
const app = getApp();
const { userApi, logApi } = require('../../utils/api.js');
const { showToast, showConfirm } = require('../../utils/util.js');

Page({
  data: {
    username: '',
    userLevel: '',
    serverUrl: '',
    userLevelMap: {
      'super_admin': '超级管理员',
      'admin': '管理员',
      'user': '普通用户'
    }
  },

  onLoad() {
    if (!app.isLoggedIn()) {
      wx.redirectTo({
        url: '/pages/login/login'
      });
      return;
    }

    this.loadUserInfo();
  },

  onShow() {
    if (app.isLoggedIn()) {
      this.loadUserInfo();
    } else {
      wx.redirectTo({
        url: '/pages/login/login'
      });
    }
  },

  // 加载用户信息
  loadUserInfo() {
    this.setData({
      username: app.globalData.username || '未登录',
      userLevel: app.globalData.userLevel || 'user',
      serverUrl: app.globalData.serverUrl
    });
  },

  // 查看历史
  viewHistory() {
    wx.switchTab({
      url: '/pages/history/history'
    });
  },

  // 查看日志
  viewLogs() {
    wx.showLoading({
      title: '加载中...'
    });

    logApi.getLog()
      .then(res => {
        wx.hideLoading();

        if (res.content) {
          wx.showModal({
            title: '操作日志',
            content: res.content || '暂无日志',
            showCancel: true,
            cancelText: '关闭',
            confirmText: '清空日志',
            success: (modalRes) => {
              if (modalRes.confirm) {
                this.clearLogs();
              }
            }
          });
        } else {
          showToast('暂无日志', 'none');
        }
      })
      .catch(err => {
        wx.hideLoading();
        showToast(err.error || '加载失败', 'none');
      });
  },

  // 清空日志
  clearLogs() {
    showConfirm('确定要清空所有日志吗？')
      .then(confirmed => {
        if (confirmed) {
          logApi.clearLog()
            .then(() => {
              showToast('日志已清空', 'success');
            })
            .catch(err => {
              showToast(err.error || '清空失败', 'none');
            });
        }
      });
  },

  // // 下载APP
  // downloadApp() {
  //   wx.showModal({
  //     title: '下载APP',
  //     content: '即将跳转到APP下载页面',
  //     success: (res) => {
  //       if (res.confirm) {
  //         // 这里可以跳转到下载页面或显示二维码
  //         showToast('功能开发中', 'none');
  //       }
  //     }
  //   });
  // },
  // 下载APP
  downloadApp() {
    const qrUrl = "https://calamari-scorecard-unsolved.ngrok-free.dev/static/qrcode_apk_example.png";
    const apkUrl = "https://calamari-scorecard-unsolved.ngrok-free.devownloads/仪态万象.apk";

    wx.showModal({
      title: '下载APP',
      content: '请扫描二维码下载APP',
      success: (res) => {
        if (res.confirm) {
          // 直接预览二维码
          wx.previewImage({
            current: qrUrl,
            urls: [qrUrl]
          });

          wx.showToast({
            title: '长按二维码下载',
            icon: 'none'
          });
        }
      }
    });
  },

  // 显示关于
  showAbout() {
    const content = [
      '仪态万象',
      '基于YOLOv11深度学习模型',
      '实现指针式仪表的智能识别与读数',
      '',
      '版本：v2.9.3',
      '© 2026 保留所有权利'
    ].join('\n');

    wx.showModal({
      title: '关于我们',
      content: content,
      showCancel: false
    });
  },

  // 显示帮助
  showHelp() {
    wx.showModal({
      title: '使用帮助',
      content: '1. 点击"检测"页面拍照或选择图片\n2. 可选填写仪表序列号\n3. 点击"开始检测"等待结果\n4. 查看检测结果，可修改读数\n5. 确认无误后点击"确认结果"\n6. 在"历史"页面查看所有记录',
      showCancel: false
    });
  },

  // 退出登录
  handleLogout() {
    showConfirm('确定要退出登录吗？')
      .then(confirmed => {
        if (confirmed) {
          // 调用退出接口（可选）
          userApi.logout()
            .catch(err => {
              console.log('退出接口调用失败:', err);
            })
            .finally(() => {
              // 清除本地登录信息
              app.clearLoginInfo();

              showToast('已退出登录', 'success');

              // 跳转到登录页
              setTimeout(() => {
                wx.reLaunch({
                  url: '/pages/login/login'
                });
              }, 1000);
            });
        }
      });
  }
});
