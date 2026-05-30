// pages/chart/chart.js
const app = getApp();
const { historyApi } = require('../../utils/api.js');
const { showToast, formatTime } = require('../../utils/util.js');

Page({
  data: {
    serial: '',
    records: [],
    loading: false,
    confirmedCount: 0,
    latestReading: '-'
  },

  onLoad(options) {
    if (!app.isLoggedIn()) {
      wx.redirectTo({
        url: '/pages/login/login'
      });
      return;
    }

    if (options.serial) {
      this.setData({
        serial: options.serial
      });
      this.loadData();
    }
  },

  // 加载数据
  loadData() {
    this.setData({ loading: true });

    historyApi.getSerialHistory(this.data.serial, 100)
      .then(res => {
        const confirmedCount = res.records.filter(r => r.is_confirmed).length;
        const latestRecord = res.records[res.records.length - 1];
        const latestReading = latestRecord ? 
          (latestRecord.reading_after || latestRecord.reading_before) : '-';

        this.setData({
          records: res.records,
          confirmedCount: confirmedCount,
          latestReading: latestReading,
          loading: false
        });

        // 绘制图表
        this.drawChart();
      })
      .catch(err => {
        console.error('加载数据失败:', err);
        showToast(err.error || '加载失败', 'none');
        this.setData({ loading: false });
      });
  },

  // 绘制图表
  drawChart() {
    const records = this.data.records;
    if (records.length === 0) return;

    const ctx = wx.createCanvasContext('lineChart');
    const width = 690; // canvas宽度（rpx转px）
    const height = 400; // canvas高度
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // 提取数据
    const readings = records.map(r => r.reading_after || r.reading_before);
    const maxReading = Math.max(...readings);
    const minReading = Math.min(...readings);
    const range = maxReading - minReading || 1;

    // 绘制背景
    ctx.setFillStyle('#ffffff');
    ctx.fillRect(0, 0, width, height);

    // 绘制网格线
    ctx.setStrokeStyle('#e0e0e0');
    ctx.setLineWidth(1);
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // 绘制Y轴刻度
    ctx.setFontSize(12);
    ctx.setFillStyle('#666666');
    ctx.setTextAlign('right');
    for (let i = 0; i <= 5; i++) {
      const value = maxReading - (range / 5) * i;
      const y = padding + (chartHeight / 5) * i;
      ctx.fillText(value.toFixed(2), padding - 5, y + 5);
    }

    // 绘制折线
    if (records.length > 0) {
      ctx.setStrokeStyle('#667eea');
      ctx.setLineWidth(2);
      ctx.beginPath();

      records.forEach((record, index) => {
        const reading = record.reading_after || record.reading_before;
        const x = padding + (chartWidth / (records.length - 1 || 1)) * index;
        const y = padding + chartHeight - ((reading - minReading) / range) * chartHeight;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // 绘制数据点
      ctx.setFillStyle('#667eea');
      records.forEach((record, index) => {
        const reading = record.reading_after || record.reading_before;
        const x = padding + (chartWidth / (records.length - 1 || 1)) * index;
        const y = padding + chartHeight - ((reading - minReading) / range) * chartHeight;

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // 绘制X轴
    ctx.setStrokeStyle('#333333');
    ctx.setLineWidth(2);
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // 绘制Y轴
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();

    ctx.draw();
  },

  // 格式化时间
  formatTime(dateStr) {
    if (!dateStr) return '';
    return formatTime(new Date(dateStr), 'MM-DD HH:mm');
  },

  // 查看详情
  viewDetail(e) {
    const taskId = e.currentTarget.dataset.taskId;
    wx.navigateTo({
      url: `/pages/detail/detail?taskId=${taskId}`
    });
  }
});
