// pages/login/login.js
const app = getApp();
const { userApi } = require('../../utils/api.js');
const { showToast, validateUsername } = require('../../utils/util.js');

Page({
  data: {
    username: '',
    password: '',
    showPassword: false,
    loading: false
  },

  onLoad(options) {
    // 如果已经登录，跳转到首页
    if (app.isLoggedIn()) {
      wx.switchTab({
        url: '/pages/index/index'
      });
    }
  },

  // 输入用户名
  onUsernameInput(e) {
    this.setData({
      username: e.detail.value
    });
  },

  // 输入密码
  onPasswordInput(e) {
    this.setData({
      password: e.detail.value
    });
  },

  // 切换密码显示
  togglePassword() {
    this.setData({
      showPassword: !this.data.showPassword
    });
  },

  // 登录
  handleLogin() {
    const { username, password } = this.data;

    // 验证输入
    if (!username) {
      showToast('请输入用户名', 'none');
      return;
    }

    if (!password) {
      showToast('请输入密码', 'none');
      return;
    }

    if (password.length < 6) {
      showToast('密码长度至少6位', 'none');
      return;
    }

    this.setData({ loading: true });

    // 调用登录接口
    userApi.login(username, password)
      .then(res => {
        if (res.message === '登录成功' && res.token) {
          // 保存登录信息
          app.setLoginInfo({
            token: res.token,
            userInfo: {
              user_id: res.user_id,
              username: res.username,
              user_level: res.user_level
            }
          });

          showToast('登录成功', 'success');

          // 跳转到首页
          setTimeout(() => {
            wx.switchTab({
              url: '/pages/index/index'
            });
          }, 1500);
        } else {
          showToast(res.error || '登录失败', 'none');
        }
      })
      .catch(err => {
        console.error('登录失败:', err);
        showToast(err.error || '登录失败，请重试', 'none');
      })
      .finally(() => {
        this.setData({ loading: false });
      });
  }
});
