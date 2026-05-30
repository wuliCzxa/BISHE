// utils/util.js - 通用工具函数

/**
 * 格式化时间
 * @param {Date} date - 时间对象
 * @param {string} format - 格式字符串
 */
function formatTime(date, format = 'YYYY-MM-DD HH:mm:ss') {
  if (!date) return '';
  
  if (typeof date === 'string') {
    date = new Date(date.replace(/-/g, '/'));
  }
  
  const year = date.getFullYear();
  const month = date.getMonth() + 1;
  const day = date.getDate();
  const hour = date.getHours();
  const minute = date.getMinutes();
  const second = date.getSeconds();

  const pad = (n) => n < 10 ? '0' + n : n;

  return format
    .replace('YYYY', year)
    .replace('MM', pad(month))
    .replace('DD', pad(day))
    .replace('HH', pad(hour))
    .replace('mm', pad(minute))
    .replace('ss', pad(second));
}

/**
 * 显示Loading
 * @param {string} title - 提示文字
 */
function showLoading(title = '加载中...') {
  wx.showLoading({
    title: title,
    mask: true
  });
}

/**
 * 隐藏Loading
 */
function hideLoading() {
  wx.hideLoading();
}

/**
 * 显示提示信息
 * @param {string} title - 提示文字
 * @param {string} icon - 图标类型
 */
function showToast(title, icon = 'success') {
  wx.showToast({
    title: title,
    icon: icon === 'success' || icon === 'error' || icon === 'loading' ? icon : 'none',
    duration: 2000
  });
}

/**
 * 显示确认对话框
 * @param {string} content - 内容
 * @param {string} title - 标题
 */
function showConfirm(content, title = '提示') {
  return new Promise((resolve) => {
    wx.showModal({
      title: title,
      content: content,
      success: (res) => {
        resolve(res.confirm);
      }
    });
  });
}

/**
 * 防抖函数
 * @param {Function} func - 执行函数
 * @param {number} wait - 等待时间
 */
function debounce(func, wait = 500) {
  let timeout;
  return function() {
    const context = this;
    const args = arguments;
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      func.apply(context, args);
    }, wait);
  };
}

/**
 * 节流函数
 * @param {Function} func - 执行函数
 * @param {number} wait - 等待时间
 */
function throttle(func, wait = 500) {
  let previous = 0;
  return function() {
    const now = Date.now();
    const context = this;
    const args = arguments;
    if (now - previous > wait) {
      func.apply(context, args);
      previous = now;
    }
  };
}

/**
 * 验证手机号
 * @param {string} phone - 手机号
 */
function validatePhone(phone) {
  return /^1[3-9]\d{9}$/.test(phone);
}

/**
 * 验证用户名
 * @param {string} username - 用户名
 */
function validateUsername(username) {
  return /^[a-zA-Z0-9_]{3,20}$/.test(username);
}

/**
 * 验证密码
 * @param {string} password - 密码
 */
function validatePassword(password) {
  return password && password.length >= 6;
}

/**
 * 获取图片完整URL
 * @param {string} path - 图片路径
 */
function getImageUrl(path) {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  const app = getApp();
  return `${app.globalData.serverUrl}/image/${encodeURIComponent(path)}`;
}

module.exports = {
  formatTime,
  showLoading,
  hideLoading,
  showToast,
  showConfirm,
  debounce,
  throttle,
  validatePhone,
  validateUsername,
  validatePassword,
  getImageUrl
};
