// // pages/history/history.js
// const app = getApp();
// const { historyApi } = require('../../utils/api.js');
// const { showToast } = require('../../utils/util.js');

// Page({
//   data: {
//     records: [],
//     page: 1,
//     size: 20,
//     total: 0,
//     loading: false,
//     hasMore: true,
//     searchSerial: '',
//     statusMap: {
//       'pending': '等待中',
//       'running': '检测中',
//       'success': '成功',
//       'failed': '失败'
//     }
//   },

//   onLoad() {
//     // 检查登录状态
//     if (!app.isLoggedIn()) {
//       wx.redirectTo({
//         url: '/pages/login/login'
//       });
//       return;
//     }

//     this.loadHistory();
//   },

//   onShow() {
//     // 每次显示时刷新数据
//     if (app.isLoggedIn()) {
//       this.refreshHistory();
//     }
//   },

//   // 下拉刷新
//   onPullDownRefresh() {
//     this.refreshHistory();
//     setTimeout(() => {
//       wx.stopPullDownRefresh();
//     }, 1000);
//   },

//   // 搜索输入
//   onSearchInput(e) {
//     this.setData({
//       searchSerial: e.detail.value
//     });
//   },

//   // 执行搜索
//   handleSearch() {
//     const serial = this.data.searchSerial.trim();
//     if (serial) {
//       // 跳转到图表页面查看序列号历史
//       wx.navigateTo({
//         url: `/pages/chart/chart?serial=${serial}`
//       });
//     } else {
//       this.refreshHistory();
//     }
//   },

//   // 刷新历史记录
//   refreshHistory() {
//     this.setData({
//       page: 1,
//       records: [],
//       hasMore: true
//     });
//     this.loadHistory();
//   },

//   // 加载历史记录
//   loadHistory() {
//     if (this.data.loading || !this.data.hasMore) {
//       return;
//     }

//     this.setData({ loading: true });

//     historyApi.getList(this.data.page, this.data.size)
//       .then(res => {
//         const newRecords = this.data.page === 1 ? res.records : [...this.data.records, ...res.records];
        
//         this.setData({
//           records: newRecords,
//           total: res.total,
//           hasMore: newRecords.length < res.total,
//           loading: false
//         });
//       })
//       .catch(err => {
//         console.error('加载历史失败:', err);
//         showToast(err.error || '加载失败', 'none');
//         this.setData({ loading: false });
//       });
//   },

//   // 加载更多
//   loadMore() {
//     if (!this.data.loading && this.data.hasMore) {
//       this.setData({
//         page: this.data.page + 1
//       });
//       this.loadHistory();
//     }
//   },

//   // 查看详情
//   viewDetail(e) {
//     const taskId = e.currentTarget.dataset.taskId;
//     wx.navigateTo({
//       url: `/pages/detail/detail?taskId=${taskId}`
//     });
//   }
// });
// pages/history/history.js
const app = getApp();
const { historyApi } = require('../../utils/api.js');
const { showToast } = require('../../utils/util.js');

Page({
  data: {
    records: [],
    page: 1,
    size: 20,
    total: 0,
    loading: false,
    hasMore: true,
    searchSerial: '',
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
      wx.redirectTo({
        url: '/pages/login/login'
      });
      return;
    }

    this.loadHistory();
  },

  onShow() {
    if (app.isLoggedIn()) {
      this.refreshHistory();
    }
  },

  // =========================
  // 下拉刷新
  // =========================
  onPullDownRefresh() {
    this.refreshHistory();
    setTimeout(() => {
      wx.stopPullDownRefresh();
    }, 800);
  },

  // =========================
  // 搜索输入
  // =========================
  onSearchInput(e) {
    this.setData({
      searchSerial: e.detail.value
    });
  },

  // =========================
  // 搜索（跳转趋势图）
  // =========================
  handleSearch() {
    const serial = this.data.searchSerial.trim();
    if (serial) {
      wx.navigateTo({
        url: `/pages/chart/chart?serial=${serial}`
      });
    } else {
      this.refreshHistory();
    }
  },

  // =========================
  // 刷新
  // =========================
  refreshHistory() {
    this.setData({
      page: 1,
      records: [],
      hasMore: true
    });
    this.loadHistory();
  },

  // =========================
  // ✅ 加载历史（核心修复）
  // =========================
  loadHistory() {
    if (this.data.loading || !this.data.hasMore) return;

    this.setData({ loading: true });

    historyApi.getList(this.data.page, this.data.size)
      .then(res => {

        // ✅ 后端返回字段适配
        const list = (res.records || []).map(item => ({
          ...item,

          // ✅ 数值安全处理（防止 null 报错）
          reading_before: item.reading_before ?? null,
          reading_after: item.reading_after ?? null,

          // ✅ 状态兜底
          detect_status: item.detect_status || 'pending',

          // ✅ 时间字符串（后端已经转了，这里保险）
          created_at: item.created_at || '',
          detected_at: item.detected_at || ''
        }));

        const newRecords = this.data.page === 1
          ? list
          : [...this.data.records, ...list];

        this.setData({
          records: newRecords,
          total: res.total || 0,
          hasMore: newRecords.length < (res.total || 0),
          loading: false
        });
      })
      .catch(err => {
        console.error('加载历史失败:', err);
        showToast(err.error || '加载失败', 'none');
        this.setData({ loading: false });
      });
  },

  // =========================
  // 加载更多
  // =========================
  loadMore() {
    if (!this.data.loading && this.data.hasMore) {
      this.setData({
        page: this.data.page + 1
      });
      this.loadHistory();
    }
  },

  // =========================
  // 查看详情
  // =========================
  viewDetail(e) {
    const taskId = e.currentTarget.dataset.taskId;

    if (!taskId) {
      showToast('task_id不存在', 'none');
      return;
    }

    wx.navigateTo({
      url: `/pages/detail/detail?taskId=${taskId}`
    });
  }
});