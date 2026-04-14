import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib
import os
import matplotlib.dates as mdates
import datetime

# 已有不同日期读数结果
file_path = 'a_reading_results_diff_day/'
# 图片保存路径
save_path = 'a_pic_results_all/'
current_time2 = datetime.datetime.now().strftime("%Y-%m-%d")
file_name = '读数变化趋势图' + current_time2 + '.png'

# 为了为每个表计创建独立的文件，我们将在保存时动态生成文件名

def makeFiledir(result_path):
    """创建输出文件夹"""
    if not os.path.exists(result_path):
        os.makedirs(result_path)


makeFiledir(save_path)

# 设置matplotlib全局使用支持中文的字体
matplotlib.rcParams['font.family'] = 'SimSun'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.unicode_minus'] = False

excel_files = glob.glob(file_path + '*.xlsx')
all_data = pd.DataFrame()

for file in excel_files:
    df = pd.read_excel(file, engine='openpyxl')
    df.drop('图片名称', axis=1, inplace=True)
    df = df[df['最终读数'] != '读数失败']
    df = df[df['检测时间'] != '未知']
    df['检测时间'] = pd.to_datetime(df['检测时间'])
    all_data = pd.concat([all_data, df])

# 为每个表计绘制一幅图
for label, grp in all_data.groupby('表计'):
    plt.figure(figsize=(20, 12))
    plt.plot(grp['检测时间'], grp['最终读数'], marker='o', linestyle='-', label=label)
    for x, y in zip(grp['检测时间'], grp['最终读数']):
        plt.text(x, y, f'{y:.4f}', color='black', fontsize=14)
    plt.xlabel('检测时间')
    plt.ylabel('最终读数')
    plt.title(f'{label}读数时间变化图')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.tight_layout()

    # 保存图表，为每个表计生成一个文件名
    file_path = os.path.join(save_path, f'{label}的读数变化趋势图.png')
    plt.savefig(file_path)
    plt.close()  # 关闭当前图表，防止图表叠加
    print(f'{label}的读数变化趋势图已绘制完成')
