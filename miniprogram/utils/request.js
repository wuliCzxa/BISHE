// // utils/request.js - 网络请求封装
// const app = getApp();

// /**
//  * 通用请求方法
//  * @param {string} url - 接口路径
//  * @param {object} options - 请求配置
//  */
// function request(url, options = {}) {
//   return new Promise((resolve, reject) => {
//     const {
//       method = 'GET',
//       data = {},
//       header = {},
//       needAuth = true
//     } = options;

//     // 构建完整URL
//     // https://69c1-2408-862e-807-c000-00-68a.ngrok-free.app/
//     const fullUrl = `${app.globalData.serverUrl}${url}`;

//     // 设置请求头
//     const requestHeader = {
//       'Content-Type': 'application/json',
//       ...header
//     };

//     // 如果需要认证，添加token
//     if (needAuth && app.globalData.token) {
//       requestHeader['Authorization'] = `Bearer ${app.globalData.token}`;
//     }

//     wx.request({
//       url: fullUrl,
//       method: method,
//       data: data,
//       header: requestHeader,
//       success: (res) => {
//         // ✅ 修改：接受 200-299 之间的所有状态码
//         if (res.statusCode >= 200 && res.statusCode < 300) {
//           resolve(res.data);
//         } else if (res.statusCode === 401) {
//           wx.showToast({
//             title: '登录已过期',
//             icon: 'none'
//           });
//           app.clearLoginInfo();
//           wx.reLaunch({
//             url: '/pages/login/login'
//           });
//           reject(new Error('未授权'));
//         } else {
//           reject(res.data);
//         }
//       },
//       fail: (error) => {
//         wx.showToast({
//           title: '网络请求失败',
//           icon: 'none'
//         });
//         reject(error);
//       }
//     });
//   });
// }

// /**
//  * 上传文件
//  * @param {string} url - 接口路径
//  * @param {string} filePath - 文件路径
//  * @param {object} formData - 额外表单数据
//  */
// function uploadFile(url, filePath, formData = {}) {
//   return new Promise((resolve, reject) => {
//     const fullUrl = `${app.globalData.serverUrl}${url}`;

//     wx.uploadFile({
//       url: fullUrl,
//       filePath: filePath,
//       name: 'file',
//       formData: formData,
//       header: {
//         'Authorization': `Bearer ${app.globalData.token || ''}`
//       },
//       success: (res) => {
//         if (res.statusCode === 200) {
//           try {
//             const data = JSON.parse(res.data);
//             resolve(data);
//           } catch (e) {
//             resolve(res.data);
//           }
//         } else if (res.statusCode === 401) {
//           wx.showToast({
//             title: '登录已过期',
//             icon: 'none'
//           });
//           app.clearLoginInfo();
//           wx.reLaunch({
//             url: '/pages/login/login'
//           });
//           reject(new Error('未授权'));
//         } else {
//           reject(res);
//         }
//       },
//       fail: (error) => {
//         wx.showToast({
//           title: '上传失败',
//           icon: 'none'
//         });
//         reject(error);
//       }
//     });
//   });
// }

// module.exports = {
//   request,
//   uploadFile
// };
// utils/request.js
const app = getApp();

/**
 * 通用请求方法
 */
function request(url, options = {}) {
  return new Promise((resolve, reject) => {
    const {
      method = 'GET',
      data = {},
      header = {},
      needAuth = true
    } = options;

    const fullUrl = `${app.globalData.serverUrl}${url}`;

    const requestHeader = {
      'Content-Type': 'application/json',

      // 🔥 关键：跳过 ngrok 提示页
      'ngrok-skip-browser-warning': 'true',

      ...header
    };

    // ✅ 自动带 token
    if (needAuth && app.globalData.token) {
      requestHeader['Authorization'] = `Bearer ${app.globalData.token}`;
    }

    wx.request({
      url: fullUrl,
      method,
      data,
      header: requestHeader,

      success: (res) => {
        console.log('[请求]', fullUrl);
        console.log('[返回]', res);

        const { statusCode, data } = res;

        // ✅ 成功状态码统一处理
        if (statusCode >= 200 && statusCode < 300) {

          // ⚠️ 后端返回 error 字段也算失败
          if (data && data.error) {
            reject(data);
            return;
          }

          resolve(data);
        }

        // ✅ token 失效
        else if (statusCode === 401) {
          handleUnauthorized();
          reject(new Error('未授权'));
        }

        // 其他错误
        else {
          reject(data || { error: '请求失败' });
        }
      },

      fail: (error) => {
        console.error('[网络错误]', error);

        wx.showToast({
          title: '网络连接失败',
          icon: 'none'
        });

        reject(error);
      }
    });
  });
}

/**
 * 上传文件
 */
function uploadFile(url, filePath, formData = {}) {
  return new Promise((resolve, reject) => {
    const fullUrl = `${app.globalData.serverUrl}${url}`;

    wx.uploadFile({
      url: fullUrl,
      filePath,
      name: 'file',
      formData,

      header: {
        'ngrok-skip-browser-warning': 'true',  // 🔥必须加

        'Authorization': `Bearer ${app.globalData.token || ''}`
      },

      success: (res) => {
        console.log('[上传]', fullUrl);
        console.log('[上传返回]', res);

        const { statusCode } = res;

        let data;
        try {
          data = JSON.parse(res.data);
        } catch (e) {
          data = res.data;
        }

        // ✅ 统一成功判断（不是只认200）
        if (statusCode >= 200 && statusCode < 300) {

          if (data && data.error) {
            reject(data);
            return;
          }

          resolve(data);
        }

        else if (statusCode === 401) {
          handleUnauthorized();
          reject(new Error('未授权'));
        }

        else {
          reject(data || { error: '上传失败' });
        }
      },

      fail: (error) => {
        console.error('[上传失败]', error);

        wx.showToast({
          title: '上传失败',
          icon: 'none'
        });

        reject(error);
      }
    });
  });
}

/**
 * ✅ 统一处理 token 失效（防止重复跳转）
 */
function handleUnauthorized() {
  if (app.globalData.isRedirecting) return;

  app.globalData.isRedirecting = true;

  wx.showToast({
    title: '登录已过期',
    icon: 'none'
  });

  app.clearLoginInfo();

  setTimeout(() => {
    wx.reLaunch({
      url: '/pages/login/login'
    });

    app.globalData.isRedirecting = false;
  }, 800);
}

module.exports = {
  request,
  uploadFile
};