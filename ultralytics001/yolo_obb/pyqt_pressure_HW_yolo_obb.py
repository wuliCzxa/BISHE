import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, \
    QFileDialog, QTextEdit, QInputDialog
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import numpy as np
import datetime
from math import sqrt
import cv2
import os
import math

class ImageDetection(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.before_imagepath = None
        self.image = None
        self.imageName = None
        self.imageName_all = None
        self.result_string = None
        # ultralytics存放路径
        self.flie_path = 'D:/毕设/ultralytics/z_pressure_HW_yolo_obb/'
        # 序号标记对照表路径
        self.file_excel_path = '序号标记对照表.xlsx'
        # 结果保存路径
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d")
        self.result_path = 'outputs-' + self.current_time + '/'
        self.makeFiledir(self.result_path)
        # 读数保存路径
        self.txt_path = self.flie_path + 'Result_pointer.txt'
        # 模型路径
        self.model_path1 = 'weight/1biaopan_all/weights/best.pt'
        self.model_path2 = 'weight/2biaopan_nolabel/weights/best.pt'
        self.model_path3 = 'weight/3biaopan_label/weights/best.pt'
        self.model_path4 = 'weight/4read/weights/best.pt'
        # 读取Excel文件
        df = pd.read_excel(self.file_excel_path, engine='openpyxl')
        # 将序号列的数据类型转换为字符串
        df['序号'] = df['序号'].astype(str)
        # Excel文件中第一列是序号，第二列是表计 提取这两列并转换为字典
        self.my_dict = pd.Series(df['表计'].values, index=df['序号']).to_dict()


    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('Pointer Detection')
        self.setGeometry(0, 0, 1800, 1250)
        # 获取屏幕的尺寸
        screen = app.primaryScreen()
        screen_rect = screen.availableGeometry()
        # 计算窗口在屏幕中央的位置
        x = (screen_rect.width() - self.width()) // 2
        y = (screen_rect.height() - self.height()) // 2
        self.move(x, y)  # 设置窗口的位置为屏幕中央

        # 创建水平布局和垂直布局
        mainLayout = QVBoxLayout()
        topLayout = QHBoxLayout()

        # 在最上方添加一行文字
        topLayout.addStretch(1)  # 在标签前添加弹性空间
        self.topLabel = QLabel('指针式仪表读数检测识别系统')
        font4 = QFont('times', 22)
        self.topLabel.setFont(font4)
        topLayout.addWidget(self.topLabel)  # 将标签添加到布局中
        # topLayout.addStretch(1)  # 在标签前添加弹性空间
        mainLayout.addLayout(topLayout)
        mainLayout.setAlignment(topLayout, Qt.AlignCenter)

        leftLayout = QVBoxLayout()
        middleLayout = QVBoxLayout()
        rightLayout = QVBoxLayout()
        contentLayout = QHBoxLayout()

        contentLayout.addLayout(leftLayout, 1)
        contentLayout.addLayout(middleLayout, 2)
        contentLayout.addLayout(rightLayout, 3)

        # 创建并设置标签来显示图片
        self.imageLabel1 = QLabel('未检测图片')
        self.imageLabel1.setAlignment(Qt.AlignCenter)
        self.imageLabel2 = QLabel('仪表盘带标签')
        self.imageLabel2.setAlignment(Qt.AlignCenter)
        self.imageLabel3 = QLabel('仪表盘')
        self.imageLabel3.setAlignment(Qt.AlignCenter)
        self.imageLabel4 = QLabel('检测拟合结果')
        self.imageLabel4.setAlignment(Qt.AlignCenter)

        # 按钮导入图片
        self.loadImageButton = QPushButton('导入图片')
        self.loadImageButton.clicked.connect(self.loadImage)
        # 按钮开始检测
        self.detectButton = QPushButton('开始检测')
        self.detectButton.clicked.connect(self.Readvalue)
        # 按钮提交取消
        self.defineButton = QPushButton('确认')
        self.defineButton.clicked.connect(self.define)
        self.modifyButton = QPushButton('修改')
        self.modifyButton.clicked.connect(self.modify)
        # 按钮清除txt内容
        self.clearButton = QPushButton('清除')
        self.clearButton.clicked.connect(self.clear)
        # 创建文本编辑器来显示文本文件内容
        self.textEdit = QTextEdit()
        # 设置定时器，定期读取文件内容
        self.timer = QTimer(self)
        self.timer.setInterval(1000)  # 设置时间间隔为1000毫秒
        self.timer.timeout.connect(self.readFile)
        self.timer.start()

        # 创建一个QLabel来显示计算后的字符串
        self.txt_label1 = QLabel(f'', self)
        self.txt_label2 = QLabel(f'', self)
        self.txt_label3 = QLabel(f'检测结果', self)
        self.txt_label4 = QLabel(f'', self)
        # 设置QLabel的字体和大小
        font = QFont('times', 12)
        font2 = QFont('times', 16)
        font3 = QFont('times', 12)
        self.txt_label1.setFont(font)   #时间
        self.txt_label2.setFont(font)   #结果1
        self.txt_label3.setFont(font2)  #检测结果
        self.txt_label4.setFont(font)   #结果2
        self.imageLabel1.setFont(font)
        self.imageLabel2.setFont(font)
        self.imageLabel3.setFont(font)
        self.imageLabel4.setFont(font)

        self.textEdit.setFont(font3)    # 文本读数结果
        self.textEdit.setStyleSheet("background-color: white; color: black;")

        self.imageLabel1.setMinimumWidth(550)
        self.imageLabel1.setMaximumWidth(550)
        self.imageLabel2.setMinimumWidth(550)
        self.imageLabel2.setMaximumWidth(550)
        self.loadImageButton.setMinimumWidth(550)
        self.loadImageButton.setMaximumWidth(550)
        self.detectButton.setMinimumWidth(550)
        self.detectButton.setMaximumWidth(550)
        self.defineButton.setMinimumWidth(550)
        self.defineButton.setMaximumWidth(550)
        self.modifyButton.setMinimumWidth(550)
        self.modifyButton.setMaximumWidth(550)
        self.textEdit.setMinimumWidth(700)
        self.textEdit.setMaximumWidth(700)

        # 将组件添加到布局中
        topLayout.addWidget(self.topLabel)

        leftLayout.addWidget(self.imageLabel1)
        leftLayout.addWidget(self.imageLabel3)
        leftLayout.addWidget(self.loadImageButton)
        leftLayout.addWidget(self.detectButton)

        middleLayout.addWidget(self.imageLabel2)
        middleLayout.addWidget(self.imageLabel4)
        middleLayout.addWidget(self.defineButton)
        middleLayout.addWidget(self.modifyButton)

        rightLayout.addWidget(self.txt_label3)
        rightLayout.addWidget(self.txt_label1)
        rightLayout.addWidget(self.txt_label4)
        rightLayout.addWidget(self.txt_label2)
        rightLayout.addWidget(self.textEdit)
        rightLayout.addWidget(self.clearButton)

        # 将左右布局添加到主布局中
        mainLayout.addLayout(leftLayout, 1)
        mainLayout.addLayout(middleLayout, 2)
        mainLayout.addLayout(rightLayout, 3)

        # 将水平布局添加到主布局中
        mainLayout.addLayout(contentLayout)
        # 设置主布局
        self.setLayout(mainLayout)

    def define(self):
        with open(self.txt_path, 'a') as file:
            file.write("The reading is correct.\n")

    def clear(self):
        with open(self.txt_path, 'w') as file:
            file.write("")

    def modify(self):
        # 创建输入对话框
        inputDialog = QInputDialog(self)
        inputDialog.setWindowTitle('Modify')
        inputDialog.setLabelText('Enter your value:')
        inputDialog.setTextValue(self.result_string)
        inputDialog.resize(260, 150)  # 设置输入对话框的大小
        # 显示对话框并等待用户响应
        ok = inputDialog.exec_()
        text = inputDialog.textValue()
        if ok:
            # 用户点击了OK，处理输入的文本
            with open(self.txt_path, 'a') as file:
                file.write("The corrected reading is:" + text + '\n')

    def loadImage(self):
        # 打开文件对话框加载图片
        imagePath, _ = QFileDialog.getOpenFileName()
        self.before_imagepath = imagePath
        self.image = cv2.imread(self.before_imagepath)
        height, weight = self.image.shape[:2]
        # print(height, weight)
        self.imageName = self.before_imagepath.split('/')[-1].split('.')[0]
        self.imageName_all = self.before_imagepath.split('/')[-1]

        if imagePath:
            pixmap = QPixmap(imagePath)
            # self.imageLabel1.setPixmap(pixmap.scaled(self.imageLabel1.size(), Qt.KeepAspectRatio))
            # scaled_pixmap = pixmap.scaled(550, int(550*height/weight))
            scaled_pixmap = pixmap.scaled(550, 550)
            self.imageLabel1.setPixmap(scaled_pixmap)


    def readFile(self):
        # 从文件中读取内容并显示在 QTextEdit 控件中
        try:
            with open(self.txt_path, 'r') as file:
                self.textEdit.setText(file.read())
        except FileNotFoundError:
            self.textEdit.setText("File not found.")

    def makeFiledir(self, result_path):
        """ 创建输出文件夹"""
        if not os.path.exists(result_path):  # 是否存在这个文件夹
            os.makedirs(result_path)  # 如果没有这个文件夹，那就创建一个


    def calculate_intersection(self, p1, p2, p3, p4):
        # 计算第一条直线的斜率和截距
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - m1 * p1[0]

        # 计算第二条直线的斜率和截距
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b2 = p3[1] - m2 * p3[0]

        # 检查斜率是否相同，即直线是否平行
        if m1 == m2:
            return None  # 直线平行或重合，没有交点

        # 计算交点的x坐标
        x = (b2 - b1) / (m1 - m2)
        # 计算交点的y坐标
        y = m1 * x + b1

        return (x, y)

    def Distances(self, a, b):
        # 返回两点间的距离
        x1, y1 = a
        x2, y2 = b
        Distances = int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        return Distances

    def GetClockAngle(self, v1, v2):
        # 2个向量模的乘积 ,返回夹角
        TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
        # 叉乘
        rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
        # 点乘
        theta = np.rad2deg(np.arccos(np.dot(v1, v2) / TheNorm))
        if rho > 0:
            return theta
        else:
            return 360 - theta

    def get_mid_point(self, x_p1, y_p1, x_p2, y_p2, x_p3, y_p3, x_p4, y_p4):
        # 定义点
        P1 = (x_p1, y_p1)
        P2 = (x_p2, y_p2)
        P3 = (x_p3, y_p3)
        P4 = (x_p4, y_p4)
        # 边及其对应的两个点
        edges = [
            (P1, P2),
            (P2, P3),
            (P3, P4),
            (P4, P1)
        ]

        # 计算边长并排序
        edges_with_length = [((p1, p2), math.hypot(p2[0] - p1[0], p2[1] - p1[1])) for (p1, p2) in edges]
        edges_with_length.sort(key=lambda x: x[1])  # 按边长升序排列

        # 获取两个最短边的中点
        def midpoint(p1, p2):
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        short_edge_1 = edges_with_length[0][0]
        short_edge_2 = edges_with_length[1][0]

        midpoint1 = midpoint(*short_edge_1)
        midpoint2 = midpoint(*short_edge_2)

        return midpoint1, midpoint2


    def Readvalue(self):
        self.result_path_img = self.result_path + self.imageName + '/'
        self.txt_path1 = self.result_path_img + 'result.txt'
        # 创建路径
        self.makeFiledir(self.result_path_img)

        # 第一步将表盘带标签裁剪出来
        yolo1 = YOLO(model=self.model_path1, task='detect')  # task非必须
        result1 = yolo1(source=self.before_imagepath, save=True, save_txt=True, save_crop=True, conf=0.7)
        save_path1 = result1[0].save_dir
        # 保存表盘带标签
        image1 = cv2.imread(save_path1 + '/crops/Instrument/' + self.imageName_all)
        cv2.imwrite(self.result_path_img + self.imageName + '_all.jpg', image1)
        # 表盘带标签
        imagepath = os.path.join(self.result_path_img, self.imageName + '_all.jpg')

        # 展示裁剪后表盘带标签图片
        pixmap = QPixmap(imagepath)
        scaled_pixmap = pixmap.scaled(550, 550)  # 将图像缩放
        self.imageLabel2.setPixmap(scaled_pixmap)

        # 第二步裁剪单表盘
        yolo2 = YOLO(model=self.model_path2, task='detect')  # task非必须
        result2 = yolo2(source=imagepath, save=True, save_txt=True, save_crop=True, conf=0.7)  # imagepath表盘带标签图像
        save_path2 = result2[0].save_dir
        # 保存单表盘
        image2 = cv2.imread(save_path2 + '/crops/Pointer/' + self.imageName + '_all.jpg')
        cv2.imwrite(self.result_path_img + self.imageName + '_biaopan.jpg', image2)

        # 展示单表盘
        pixmap = QPixmap(self.result_path_img + self.imageName + '_biaopan.jpg')
        scaled_pixmap = pixmap.scaled(550, 550)  # 将图像缩放
        self.imageLabel3.setPixmap(scaled_pixmap)

        # 第三步裁剪标签
        yolo3 = YOLO(model=self.model_path3, task='detect')  # task非必须
        result3 = yolo3(source=imagepath, save=True, save_txt=True, save_crop=True, conf=0.7)  # imagepath表盘带标签图像
        save_path3 = result3[0].save_dir
        # 保存标签
        image3 = cv2.imread(save_path3 + '/crops/Label/' + self.imageName + '_all.jpg')
        cv2.imwrite(self.result_path_img + self.imageName + '_biaoqian.jpg', image3)

        # 第四步识别标签
        yolo4 = YOLO(model=self.model_path4, task='detect')  # task非必须
        result4 = yolo4(source=self.result_path_img + self.imageName + '_biaoqian.jpg', save=True, save_txt=True, conf=0.7)
        save_path4 = result4[0].save_dir
        # txt文件路径
        label_path4 = save_path4 + '/' + 'labels'
        # print("label_path4:", label_path4)

        # 第五步识别读数
        yolo5 = YOLO(model=self.model_path4, task='detect')  # task非必须
        result5 = yolo5(source=self.result_path_img + self.imageName + '_biaopan.jpg', save=True, save_txt=True, conf=0.7)
        save_path5 = result5[0].save_dir
        # txt文件路径
        label_path5 = save_path5 + '/' + 'labels'
        # print("label_path5:", label_path5)

        # 读取图像
        image = cv2.imread(save_path5 + '/' + self.imageName + '_biaopan.jpg')
        # 获取图像的高度和宽度
        height, width = image.shape[:2]
        # print(height, width)

        # txt文件
        file_path1 = os.path.join(label_path5, self.imageName + '_biaopan.txt')
        # 初始化一个空列表，用于存储每一行的数据
        rows_as_lists = []
        with open(file_path1, 'r') as f:
            for line in f:
                # 使用split()方法将行分割成列表，这里假设数据之间是用空格分隔的
                row_list = line.strip().split()  # strip()用于移除字符串头尾指定的字符（默认为空格）
                # 将得到的列表添加到rows_as_lists中
                rows_as_lists.append(row_list)

        # 先按第一列排序，再按第二列排序
        sorted_rows = sorted(rows_as_lists, key=lambda x: (float(x[0]), float(x[1])))
        # print(sorted_rows)

        # 起始刻度线坐标
        x_s1 = width * float(sorted_rows[0][7])
        y_s1 = height * float(sorted_rows[0][8])
        x_s2 = width * float(sorted_rows[0][1])
        y_s2 = height * float(sorted_rows[0][2])
        x_s3 = width * float(sorted_rows[0][3])
        y_s3 = height * float(sorted_rows[0][4])
        x_s4 = width * float(sorted_rows[0][5])
        y_s4 = height * float(sorted_rows[0][6])
        # 拟合起始刻度线坐标
        x_sf = (x_s1 + x_s2 + x_s3 + x_s4) / 4
        y_sf = (y_s1 + y_s2 + y_s3 + y_s4) / 4

        # 终止刻度线坐标
        x_e1 = width * float(sorted_rows[1][1])
        y_e1 = height * float(sorted_rows[1][2])
        x_e2 = width * float(sorted_rows[1][3])
        y_e2 = height * float(sorted_rows[1][4])
        x_e3 = width * float(sorted_rows[1][5])
        y_e3 = height * float(sorted_rows[1][6])
        x_e4 = width * float(sorted_rows[1][7])
        y_e4 = height * float(sorted_rows[1][8])
        # 拟合终止刻度线坐标
        x_ef = (x_e1 + x_e2 + x_e3 + x_e4) / 4
        y_ef = (y_e1 + y_e2 + y_e3 + y_e4) / 4

        # 指针端点坐标
        x_p1 = width * float(sorted_rows[3][7])
        y_p1 = height * float(sorted_rows[3][8])
        x_p2 = width * float(sorted_rows[3][1])
        y_p2 = height * float(sorted_rows[3][2])
        x_p3 = width * float(sorted_rows[3][3])
        y_p3 = height * float(sorted_rows[3][4])
        x_p4 = width * float(sorted_rows[3][5])
        y_p4 = height * float(sorted_rows[3][6])
        # 拟合指针端点坐标
        x_pf = (x_p1 + x_p2 + x_p3 + x_p4) / 4
        y_pf = (y_p1 + y_p2 + y_p3 + y_p4) / 4

        # 中间刻度线坐标
        x_z1 = width * float(sorted_rows[2][1])
        y_z1 = height * float(sorted_rows[2][2])
        x_z2 = width * float(sorted_rows[2][3])
        y_z2 = height * float(sorted_rows[2][4])
        x_z3 = width * float(sorted_rows[2][5])
        y_z3 = height * float(sorted_rows[2][6])
        x_z4 = width * float(sorted_rows[2][7])
        y_z4 = height * float(sorted_rows[2][8])
        # 拟合中间刻度线坐标
        x_zf = (x_z1 + x_z2 + x_z3 + x_z4) / 4
        y_zf = (y_z1 + y_z2 + y_z3 + y_z4) / 4


        # 计算圆心坐标
        # 两个短边中点坐标
        (x_ss1, y_ss1), (x_ss2, y_ss2) = self.get_mid_point(x_s1, y_s1, x_s2, y_s2, x_s3, y_s3, x_s4, y_s4)
        p1 = (x_ss1, y_ss1)
        p2 = (x_ss2, y_ss2)

        # 两个短边中点坐标
        (x_ee1, y_ee1), (x_ee2, y_ee2) = self.get_mid_point(x_e1, y_e1, x_e2, y_e2, x_e3, y_e3, x_e4, y_e4)
        p3 = (x_ee1, y_ee1)
        p4 = (x_ee2, y_ee2)

        # 两个短边中点坐标
        (x_pp1, y_pp1), (x_pp2, y_pp2) = self.get_mid_point(x_p1, y_p1, x_p2, y_p2, x_p3, y_p3, x_p4, y_p4)
        p5 = (x_pp1, y_pp1)
        p6 = (x_pp2, y_pp2)

        # 两个短边中点坐标
        (x_zz1, y_zz1), (x_zz2, y_zz2) = self.get_mid_point(x_z1, y_z1, x_z2, y_z2, x_z3, y_z3, x_z4, y_z4)
        p7 = (x_zz1, y_zz1)
        p8 = (x_zz2, y_zz2)

        # 计算开始刻度线、结束刻度线、中间刻度线交点坐标
        intersection_se = self.calculate_intersection(p1, p2, p3, p4)
        intersection_sz = self.calculate_intersection(p1, p2, p7, p8)
        intersection_ez = self.calculate_intersection(p3, p4, p7, p8)

        d11 = self.Distances(intersection_se, intersection_sz)
        d22 = self.Distances(intersection_se, intersection_ez)
        d33 = self.Distances(intersection_sz, intersection_ez)
        d44 = min(d11, d22, d33)

        # 由开始刻度线、结束刻度线、中间刻度线计算的圆心坐标1
        if d44 == d11:
            c_xx = (intersection_se[0] + intersection_sz[0]) / 2
            c_yy = (intersection_se[1] + intersection_sz[1]) / 2
        elif d44 == d22:
            c_xx = (intersection_se[0] + intersection_ez[0]) / 2
            c_yy = (intersection_se[1] + intersection_ez[1]) / 2
        elif d44 == d33:
            c_xx = (intersection_sz[0] + intersection_ez[0]) / 2
            c_yy = (intersection_sz[1] + intersection_ez[1]) / 2

        # 计算计算开始刻度线、结束刻度线、指针交点坐标
        intersection_se = self.calculate_intersection(p1, p2, p3, p4)
        intersection_sp = self.calculate_intersection(p1, p2, p5, p6)
        intersection_ep = self.calculate_intersection(p3, p4, p5, p6)

        d1 = self.Distances(intersection_se, intersection_sp)
        d2 = self.Distances(intersection_se, intersection_ep)
        d3 = self.Distances(intersection_sp, intersection_ep)
        d4 = min(d1, d2, d3)

        # 由开始刻度线、结束刻度线、指针计算的圆心坐标2
        if d4 == d1:
            c_x = (intersection_se[0] + intersection_sp[0]) / 2
            c_y = (intersection_se[1] + intersection_sp[1]) / 2
        elif d4 == d2:
            c_x = (intersection_se[0] + intersection_ep[0]) / 2
            c_y = (intersection_se[1] + intersection_ep[1]) / 2
        elif d4 == d3:
            c_x = (intersection_sp[0] + intersection_ep[0]) / 2
            c_y = (intersection_sp[1] + intersection_ep[1]) / 2

        # 定义两个点的坐标
        point1 = (int(intersection_se[0]), int(intersection_se[1]))
        point2 = (int(intersection_sp[0]), int(intersection_sp[1]))
        point3 = (int(intersection_ep[0]), int(intersection_ep[1]))
        point4 = (int(c_x), int(c_y))
        point5 = (int(x_zf), int(y_zf))
        point6 = (int(c_xx), int(c_yy))

        # 在图像上标记点，这里设置圆的半径为5，颜色为红色(0, 0, 255)，线宽为-1表示填充圆
        cv2.circle(image, point1, 5, (0, 0, 255), -1)  # 开始 结束
        cv2.circle(image, point2, 5, (0, 0, 120), -1)  # 开始 指针
        cv2.circle(image, point3, 5, (0, 0, 0), -1)  # 结束 指针
        cv2.circle(image, point4, 5, (0, 0, 255), -1)
        cv2.circle(image, point5, 5, (0, 0, 255), -1)
        cv2.circle(image, point6, 5, (0, 0, 255), -1)

        # 保存图像
        cv2.imwrite(self.result_path_img + self.imageName + "_fitting.jpg", image)

        # 展示拟合图片
        pixmap = QPixmap(self.result_path_img + self.imageName + "_fitting.jpg", image)
        scaled_pixmap = pixmap.scaled(550, 550)  # 将图像缩放
        self.imageLabel4.setPixmap(scaled_pixmap)

        # 计算修正前读数
        # 计算零刻度与指针端点所成角度
        v1 = [x_sf - c_x, y_sf - c_y]
        v2 = [x_pf - c_x, y_pf - c_y]
        theta1 = self.GetClockAngle(v1, v2)
        # 计算零刻度与最终刻度所成角度
        v3 = [x_sf - c_x, y_sf - c_y]
        v4 = [x_ef - c_x, y_ef - c_y]
        theta2 = self.GetClockAngle(v3, v4)
        # 计算修正前最终读数
        readValue1 = theta1 / theta2 * 1 - 0.1
        # print(readValue1)

        # 计算修正后读数
        # 最终圆心坐标
        cx = (c_x + c_xx) / 2
        cy = (c_y + c_yy) / 2
        # 计算零刻度与指针端点所成角度
        v5 = [x_sf - cx, y_sf - cy]
        v6 = [x_pf - cx, y_pf - cy]
        theta3 = self.GetClockAngle(v5, v6)

        # 计算零刻度与最终刻度所成角度
        v7 = [x_sf - cx, y_sf - cy]
        v8 = [x_ef - cx, y_ef - cy]
        theta4 = self.GetClockAngle(v7, v8)

        # 计算最终读数
        readValue2 = theta3 / theta4 * 1 - 0.1
        # print(readValue2)

        # 计算零刻度与0.4所成角度
        v9 = [x_sf - cx, y_sf - cy]
        v10 = [x_zf - cx, y_zf - cy]
        theta5 = self.GetClockAngle(v9, v10)
        # 计算最终读数2
        readValue3 = theta5 / theta4 * 1 - 0.1
        # print(readValue3)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.txt_label1.setText(current_time)
        self.txt_label2.setText('修正前读数为: ' + str(readValue1))
        self.txt_label4.setText('修正后读数为: ' + str(readValue2 + (0.4 - readValue3) / 2))
        # 读取txt文件
        file_path2 = os.path.join(label_path4, self.imageName + '_biaoqian.txt')
        with open(file_path2, 'r') as file:
            for line in file:
                first_column = line.split()[0]  # 读取每行的第一列
                with open(self.txt_path, 'a', encoding='GBK') as file:
                    file.write(current_time + '\n')
                    file.write(self.imageName + ' ')
                    file.write(self.my_dict[first_column] + '\n')
                    file.write('修正前读数为' + str(readValue1) + '\n')
                    file.write('修正后读数为' + str(readValue2 + (0.4 - readValue3) / 2) + '\n')
                with open(self.txt_path1, 'a', encoding='GBK') as file:
                    file.write(current_time + '\n')
                    file.write(self.imageName + ' ')
                    file.write(self.my_dict[first_column] + '\n')
                    file.write('修正前读数为' + str(readValue1) + '\n')
                    file.write('修正后读数为' + str(readValue2 + (0.4 - readValue3) / 2) + '\n')


# 创建应用程序和窗口实例，并运行应用程序
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageDetection()
    ex.show()
    sys.exit(app.exec_())
