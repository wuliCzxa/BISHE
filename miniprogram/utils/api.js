// utils/api.js
const { request, uploadFile } = require('./request.js');

/**
 * =========================
 * 用户API
 * =========================
 */
const userApi = {

  login(username, password) {
    return request('/api/wechat/login', {
      method: 'POST',
      data: { username, password },
      needAuth: false
    });
  },

  register(username, password) {
    return request('/api/wechat/register', {
      method: 'POST',
      data: { username, password },
      needAuth: false
    });
  },

  logout() {
    return request('/api/wechat/logout', {
      method: 'POST'
    });
  }
};


/**
 * =========================
 * 检测API（核心模块）
 * =========================
 */
const detectApi = {

  /**
   * ✅ 上传图片（支持序列号）
   */
  upload(filePath, serialNumber = '') {
    return uploadFile('/api/wechat/upload', filePath, {
      serial_number: serialNumber
    });
  },

  /**
   * ✅ 上传 base64（摄像头流）
   */
  uploadBase64(base64) {
    return request('/api/wechat/upload_base64', {
      method: 'POST',
      data: { image: base64 }
    });
  },

  /**
   * ✅ 开始检测
   */
  startDetect(taskId) {
    return request('/api/wechat/detect', {
      method: 'POST',
      data: { task_id: taskId }
    });
  },

  /**
   * ✅ 轮询检测状态（统一数据结构）
   */
  poll(taskId) {
    return request(`/api/wechat/poll/${taskId}`, {
      method: 'GET'
    }).then(res => {

      // 🔥 统一前端字段（关键！）
      return {
        ...res,

        task_id: taskId,

        // 图片字段统一
        obb_img_path: res.img_obb,
        dial_img_path: res.img_dial,
        label_img_path: res.img_fitting,

        // 状态兜底
        status: res.status || 'pending',

        // 数值安全
        reading_before: res.reading_before ?? null,
        reading_after: res.reading_after ?? null
      };
    });
  },

  /**
   * ✅ 确认结果
   */
  confirm(taskId) {
    return request('/api/wechat/confirm', {
      method: 'POST',
      data: { task_id: taskId }
    });
  },

  /**
   * ✅ 修改读数
   */
  modify(taskId, value) {
    return request('/api/wechat/modify', {
      method: 'POST',
      data: { task_id: taskId, value }
    });
  }
};


/**
 * =========================
 * 历史记录API
 * =========================
 */
const historyApi = {

  getList(page = 1, size = 20) {
    return request('/api/wechat/history', {
      method: 'GET',
      data: { page, size }
    }).then(res => {

      // ✅ 统一数据结构
      const records = (res.records || []).map(item => ({
        ...item,
        reading_before: item.reading_before ?? null,
        reading_after: item.reading_after ?? null,
        detect_status: item.detect_status || 'pending'
      }));

      return {
        ...res,
        records
      };
    });
  },

  getSerialHistory(serial, limit = 60) {
    return request('/api/wechat/serial_history', {
      method: 'GET',
      data: { serial, limit }
    });
  }
};


/**
 * =========================
 * 日志API
 * =========================
 */
const logApi = {

  getLog() {
    return request('/api/wechat/get_log', {
      method: 'GET'
    });
  },

  clearLog() {
    return request('/api/wechat/clear', {
      method: 'POST'
    });
  }
};


module.exports = {
  userApi,
  detectApi,
  historyApi,
  logApi
};