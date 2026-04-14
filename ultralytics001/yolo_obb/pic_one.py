import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib
import os
import matplotlib.dates as mdates
import datetime

# 已有不同日期读数结果
file_path = 'a_reading_results_show_pic_one/'
save_path = 'a_pic_results_one/'
current_time2 = datetime.datetime.now().strftime("%Y-%m-%d")
file_name = '读数变化趋势图' + current_time2 + '.png'

def makeFiledir(result_path):
    """ 创建输出文件夹"""
    if not os.path.exists(result_path):  # 是否存在这个文件夹
        os.makedirs(result_path)  # 如果没有这个文件夹，那就创建一个

makeFiledir(save_path)

# 设置matplotlib全局使用支持中文的字体
matplotlib.rcParams['font.family'] = 'SimSun'  # 例如使用黑体SimHei
matplotlib.rcParams['font.size'] = 20  # 可以根据需要调整字体大小
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 获取所有Excel文件的路径
excel_files = glob.glob(file_path + '*.xlsx')
# print(excel_files)
# 初始化一个空的DataFrame来汇总所有数据
all_data = pd.DataFrame()

for file in excel_files:
    # 读取Excel文件
    df = pd.read_excel(file, engine='openpyxl')

    # 删除“图片名称”列
    df.drop('图片名称', axis=1, inplace=True)

    # 处理“读数失败”和“未知”
    df = df[df['最终读数'] != '读数失败']
    df = df[df['检测时间'] != '未知']

    # 将“检测时间”转换为日期类型
    df['检测时间'] = pd.to_datetime(df['检测时间'])

    # 将处理后的数据添加到汇总DataFrame中
    all_data = pd.concat([all_data, df])

# 绘制趋势图
plt.figure(figsize=(20, 12))
# print(all_data)
for label, grp in all_data.groupby('表计'):
    plt.plot(grp['检测时间'], grp['最终读数'], marker='o', linestyle='-', label=label)
    # 在每个数据点旁边添加读数
    for x, y in zip(grp['检测时间'], grp['最终读数']):
        plt.text(x, y, f'{y:.4f}', color='black', fontsize=14)

plt.xlabel('检测时间')
plt.ylabel('最终读数')
plt.title('表计读数变化图')
plt.legend()

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 设置x轴主要刻度的显示格式为月-日
plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # 设置x轴主要刻度间隔为每天
# plt.gcf().autofmt_xdate()  # 自动旋转日期标记以防它们重叠

# plt.xticks(rotation=45)
plt.tight_layout()

# 完整的文件路径
file_path = os.path.join(save_path, file_name)
# 保存图表到文件
plt.savefig(file_path)
print('图片已绘制完成')
