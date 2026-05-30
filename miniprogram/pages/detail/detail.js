// // pages/detail/detail.js
// const app = getApp();
// const { detectApi } = require('../../utils/api.js');
// const { showToast, showLoading, hideLoading, showConfirm, getImageUrl } = require('../../utils/util.js');

// Page({
//   data: {
//     taskId: '',
//     detail: null,
//     loading: false,
//     statusMap: {
//       'pending': '等待中',
//       'running': '检测中',
//       'success': '成功',
//       'failed': '失败'
//     }
//   },

//   onLoad(options) {
//     if (!app.isLoggedIn()) {
//       wx.redirectTo({
//         url: '/pages/login/login'
//       });
//       return;
//     }

//     if (options.taskId) {
//       this.setData({
//         taskId: options.taskId
//       });
//       this.loadDetail();
//     }
//   },

//   // 加载详情
//   // loadDetail() {
//   //   this.setData({ loading: true });

//   //   detectApi.getResult(this.data.taskId)
//   //     .then(res => {
//   //       this.setData({
//   //         detail: res,
//   //         loading: false
//   //       });
//   //     })
//   //     .catch(err => {
//   //       console.error('加载详情失败:', err);
//   //       showToast(err.error || '加载失败', 'none');
//   //       this.setData({ loading: false });
//   //     });
//   // },
//   loadDetail() {
//     this.setData({ loading: true });
  
//     detectApi.poll(this.data.taskId)
//       .then(res => {
//         this.setData({
//           detail: {
//             ...res,
//             task_id: this.data.taskId
//           },
//           loading: false
//         });
//       })
//       .catch(err => {
//         console.error('加载详情失败:', err);
//         showToast(err.error || '加载失败', 'none');
//         this.setData({ loading: false });
//       });
//   },

//   // 修改读数
//   modifyReading() {
//     wx.showModal({
//       title: '修改读数',
//       editable: true,
//       placeholderText: '请输入新的读数值',
//       content: this.data.detail.reading_after?.toString() || this.data.detail.reading_before?.toString() || '',
//       success: (res) => {
//         if (res.confirm && res.content) {
//           const value = parseFloat(res.content);
//           if (isNaN(value)) {
//             showToast('请输入有效的数字', 'none');
//             return;
//           }

//           showLoading('修改中...');
//           detectApi.modify(this.data.taskId, value)
//             .then(() => {
//               hideLoading();
//               showToast('修改成功', 'success');
              
//               // 重新加载详情
//               this.loadDetail();
//             })
//             .catch(err => {
//               hideLoading();
//               showToast(err.error || '修改失败', 'none');
//             });
//         }
//       }
//     });
//   },

//   // 确认结果
//   confirmResult() {
//     showConfirm('确认当前检测结果？')
//       .then(confirmed => {
//         if (confirmed) {
//           showLoading('确认中...');
//           detectApi.confirm(this.data.taskId)
//             .then(() => {
//               hideLoading();
//               showToast('已确认', 'success');
              
//               // 重新加载详情
//               this.loadDetail();
//             })
//             .catch(err => {
//               hideLoading();
//               showToast(err.error || '确认失败', 'none');
//             });
//         }
//       });
//   },

//   // 查看趋势图
//   viewChart() {
//     if (this.data.detail && this.data.detail.serial_number) {
//       wx.navigateTo({
//         url: `/pages/chart/chart?serial=${this.data.detail.serial_number}`
//       });
//     }
//   },

//   // 预览图片
//   previewImage(e) {
//     const url = e.currentTarget.dataset.url;
//     const urls = [getImageUrl(this.data.detail.original_img_path)];
    
//     if (this.data.detail.dial_img_path) {
//       urls.push(getImageUrl(this.data.detail.dial_img_path));
//     }
//     if (this.data.detail.label_img_path) {
//       urls.push(getImageUrl(this.data.detail.label_img_path));
//     }
//     if (this.data.detail.obb_img_path) {
//       urls.push(getImageUrl(this.data.detail.obb_img_path));
//     }

//     wx.previewImage({
//       current: url,
//       urls: urls
//     });
//   },

//   // 获取图片URL
//   getImageUrl(path) {
//     return getImageUrl(path);
//   }
// });
// pages/detail/detail.js
const app = getApp();
const { detectApi } = require('../../utils/api.js');
const { showToast, showLoading, hideLoading, showConfirm, getImageUrl } = require('../../utils/util.js');

Page({
  data: {
    taskId: '',
    detail: null,
    loading: false,
    timer: null, // ✅ 轮询定时器
    statusMap: {
      'pending': '等待中',
      'running': '检测中',
      'success': '成功',
      'failed': '失败',
      'uploaded': '已上传'
    }
  },

  onLoad(options) {
    if (!app.isLoggedIn()) {
      wx.redirectTo({
        url: '/pages/login/login'
      });
      return;
    }

    if (options.taskId) {
      this.setData({
        taskId: options.taskId
      });

      // ✅ 自动开始轮询
      this.startPolling();
    }
  },

  onUnload() {
    this.stopPolling(); // 防止内存泄漏
  },

  // =========================
  // ✅ 开始轮询
  // =========================
  startPolling() {
    this.loadDetail();

    const timer = setInterval(() => {
      this.loadDetail();
    }, 1500);

    this.setData({ timer });
  },

  stopPolling() {
    if (this.data.timer) {
      clearInterval(this.data.timer);
    }
  },

  // =========================
  // ✅ 加载详情（核心修复）
  // =========================
  loadDetail() {
    detectApi.poll(this.data.taskId)
      .then(res => {

        // ✅ 图片路径适配（后端返回的是 img_xxx）
        const detail = {
          ...res,
          task_id: this.data.taskId,

          // ✅ 统一字段（适配你原页面）
          original_img_path: res.img_obb,   // 原图你后端没返回，只能先用这个
          obb_img_path: res.img_obb,
          dial_img_path: res.img_dial,
          label_img_path: res.img_fitting
        };

        this.setData({
          detail,
          loading: false
        });

        // ✅ 如果检测完成，停止轮询
        if (res.status === 'success' || res.status === 'failed') {
          this.stopPolling();
        }

      })
      .catch(err => {
        console.error('加载详情失败:', err);
        showToast(err.error || '加载失败', 'none');
      });
  },

  // =========================
  // 修改读数
  // =========================
  modifyReading() {
    const d = this.data.detail || {};

    wx.showModal({
      title: '修改读数',
      editable: true,
      placeholderText: '请输入新的读数值',
      content: d.reading_after?.toString() || d.reading_before?.toString() || '',
      success: (res) => {
        if (res.confirm && res.content) {
          const value = parseFloat(res.content);
          if (isNaN(value)) {
            showToast('请输入有效的数字', 'none');
            return;
          }

          showLoading('修改中...');
          detectApi.modify(this.data.taskId, value)
            .then(() => {
              hideLoading();
              showToast('修改成功', 'success');
              this.loadDetail();
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
  // 确认结果
  // =========================
  confirmResult() {
    showConfirm('确认当前检测结果？')
      .then(confirmed => {
        if (confirmed) {
          showLoading('确认中...');
          detectApi.confirm(this.data.taskId)
            .then(() => {
              hideLoading();
              showToast('已确认', 'success');
              this.loadDetail();
            })
            .catch(err => {
              hideLoading();
              showToast(err.error || '确认失败', 'none');
            });
        }
      });
  },

  // =========================
  // 查看趋势图
  // =========================
  viewChart() {
    if (this.data.detail && this.data.detail.serial_number) {
      wx.navigateTo({
        url: `/pages/chart/chart?serial=${this.data.detail.serial_number}`
      });
    }
  },

  // =========================
  // 预览图片（✅ 已修复）
  // =========================
  previewImage(e) {
    const url = e.currentTarget.dataset.url;

    const urls = [];

    if (this.data.detail.obb_img_path) {
      urls.push(getImageUrl(this.data.detail.obb_img_path));
    }
    if (this.data.detail.dial_img_path) {
      urls.push(getImageUrl(this.data.detail.dial_img_path));
    }
    if (this.data.detail.label_img_path) {
      urls.push(getImageUrl(this.data.detail.label_img_path));
    }

    if (urls.length === 0) {
      showToast('暂无图片', 'none');
      return;
    }

    wx.previewImage({
      current: url,
      urls: urls
    });
  },

  getImageUrl(path) {
    return getImageUrl(path);
  }
});