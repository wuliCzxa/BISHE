// pages/register/register.js
const { userApi } = require('../../utils/api.js');
const { showToast, validateUsername, validatePassword } = require('../../utils/util.js');

Page({
  data: {
    username: '',
    password: '',
    confirmPassword: '',
    showPassword: false,
    showConfirmPassword: false,
    loading: false
  },

  onUsernameInput(e) {
    this.setData({
      username: e.detail.value
    });
  },

  onPasswordInput(e) {
    this.setData({
      password: e.detail.value
    });
  },

  onConfirmPasswordInput(e) {
    this.setData({
      confirmPassword: e.detail.value
    });
  },

  togglePassword() {
    this.setData({
      showPassword: !this.data.showPassword
    });
  },

  toggleConfirmPassword() {
    this.setData({
      showConfirmPassword: !this.data.showConfirmPassword
    });
  },

  handleRegister() {
    const { username, password, confirmPassword } = this.data;

    // 验证输入
    if (!username) {
      showToast('请输入用户名', 'none');
      return;
    }

    if (!validateUsername(username)) {
      showToast('用户名格式不正确（3-20位字母数字下划线）', 'none');
      return;
    }

    if (!password) {
      showToast('请输入密码', 'none');
      return;
    }

    if (!validatePassword(password)) {
      showToast('密码长度至少6位', 'none');
      return;
    }

    if (password !== confirmPassword) {
      showToast('两次输入的密码不一致', 'none');
      return;
    }

    this.setData({ loading: true });

    // 调用注册接口
    userApi.register(username, password)
      .then(res => {
        if (res.message === '注册成功') {
          showToast('注册成功，请登录', 'success');

          // 跳转到登录页
          setTimeout(() => {
            wx.navigateBack();
          }, 1500);
        } else {
          showToast(res.error || '注册失败', 'none');
        }
      })
      .catch(err => {
        console.error('注册失败:', err);
        showToast(err.error || '注册失败，请重试', 'none');
      })
      .finally(() => {
        this.setData({ loading: false });
      });
  }
});
