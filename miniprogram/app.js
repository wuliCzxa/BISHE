// app.js - 小程序主入口
App({
  globalData: {
    // 服务器地址配置 - 修改为你的Flask服务器地址
    serverUrl: 'https://calamari-scorecard-unsolved.ngrok-free.dev // 生产环境
    // serverUrl: 'http://localhost:5000',  // 本地开发（需要在微信开发者工具中配置不校验域名）

    userInfo: null,
    token: null,
    userId: null,
    username: null,
    userLevel: null
  },

  onLaunch() {
    // 小程序启动时检查登录状态
    this.checkLoginStatus();
  },

  // 检查登录状态
  checkLoginStatus() {
    const token = wx.getStorageSync('token');
    const userInfo = wx.getStorageSync('userInfo');

    if (token && userInfo) {
      this.globalData.token = token;
      this.globalData.userInfo = userInfo;
      this.globalData.userId = userInfo.user_id;
      this.globalData.username = userInfo.username;
      this.globalData.userLevel = userInfo.user_level;
      return true;
    }
    return false;
  },

  // 设置登录信息
  setLoginInfo(data) {
    this.globalData.token = data.token;
    this.globalData.userInfo = data.userInfo;
    this.globalData.userId = data.userInfo.user_id;
    this.globalData.username = data.userInfo.username;
    this.globalData.userLevel = data.userInfo.user_level;

    wx.setStorageSync('token', data.token);
    wx.setStorageSync('userInfo', data.userInfo);
  },

  // 清除登录信息
  clearLoginInfo() {
    this.globalData.token = null;
    this.globalData.userInfo = null;
    this.globalData.userId = null;
    this.globalData.username = null;
    this.globalData.userLevel = null;

    wx.removeStorageSync('token');
    wx.removeStorageSync('userInfo');
  },

  // 检查是否登录
  isLoggedIn() {
    return !!this.globalData.token;
  }
});
