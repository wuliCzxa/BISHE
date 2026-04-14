import math
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, \
    QFileDialog, QTextEdit, QInputDialog
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import numpy as np
import datetime
from math import sqrt
import cv2
import random
import os

class ImageDetection(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.before_imagepath = None
        self.later_imagepath = None
        self.result_string = None
        self.flie_path = 'C:/Users/xxa/Desktop/ultralytics/z_pressure_pointer_4yolo_pose/'  # z_pressure_pointer_4yolo_pose绝对路径
        self.model_path1 = 'z_weight_p/weights/best.pt'    # 权重文件路径
        self.model_path2 = 'z_weight_k/weights/best.pt'    # 权重文件路径
        self.txt_path = self.flie_path + 'Result_pressure_pointer_4yolo_pose.txt'  # 读数保存路径
        self.imagepath = self.before_imagepath
        self.image = None
        self.outputPath = ('outputs/')
        self.imageName = None
        self.r = None  # 半径
        self.divisionValue = None  # 分度值
        self.lineSet = None
        self.makeFiledir()

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('Pointer Detection')
        self.setGeometry(0, 0, 1900, 1050)
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
        self.topLabel = QLabel('指针式仪表读数识别检测系统')
        font4 = QFont('times', 26)
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
        self.imageLabel2 = QLabel('检测后图片')
        self.imageLabel2.setAlignment(Qt.AlignCenter)

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
        self.txt_label5 = QLabel(f'', self)
        # 设置QLabel的字体和大小
        font = QFont('times', 16)
        font2 = QFont('times', 18)
        font3 = QFont('times', 12)
        self.txt_label1.setFont(font)  # 时间
        self.txt_label2.setFont(font)  # 结果1
        self.txt_label4.setFont(font)  # 结果2
        self.txt_label5.setFont(font)  # 结果3
        self.imageLabel1.setFont(font)
        self.imageLabel2.setFont(font)
        self.txt_label3.setFont(font2)  # 检测结果
        self.textEdit.setFont(font3)
        self.textEdit.setStyleSheet("background-color: white; color: black;")

        self.imageLabel1.setMinimumWidth(600)
        self.imageLabel1.setMaximumWidth(600)
        self.imageLabel2.setMinimumWidth(600)
        self.imageLabel2.setMaximumWidth(600)
        self.loadImageButton.setMinimumWidth(600)
        self.loadImageButton.setMaximumWidth(600)
        self.detectButton.setMinimumWidth(600)
        self.detectButton.setMaximumWidth(600)
        self.defineButton.setMinimumWidth(600)
        self.defineButton.setMaximumWidth(600)
        self.modifyButton.setMinimumWidth(600)
        self.modifyButton.setMaximumWidth(600)
        self.textEdit.setMinimumWidth(700)
        self.textEdit.setMaximumWidth(700)

        # 将组件添加到布局中
        topLayout.addWidget(self.topLabel)

        leftLayout.addWidget(self.imageLabel1)
        leftLayout.addWidget(self.loadImageButton)
        leftLayout.addWidget(self.detectButton)

        middleLayout.addWidget(self.imageLabel2)
        middleLayout.addWidget(self.defineButton)
        middleLayout.addWidget(self.modifyButton)

        rightLayout.addWidget(self.txt_label3)
        rightLayout.addWidget(self.txt_label1)
        rightLayout.addWidget(self.txt_label4)
        rightLayout.addWidget(self.txt_label5)
        rightLayout.addWidget(self.txt_label2)
        rightLayout.addWidget(self.textEdit)
        rightLayout.addWidget(self.clearButton)

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

    def Distances(self, a, b):
        # 返回两点间的距离
        x1, y1 = a
        x2, y2 = b
        Distances = int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        return Distances

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
        if imagePath:
            pixmap = QPixmap(imagePath)
            # self.imageLabel1.setPixmap(pixmap.scaled(self.imageLabel1.size(), Qt.KeepAspectRatio))
            self.image = cv2.imread(self.before_imagepath)
            self.image_h, self.image_w, _ = self.image.shape
            self.imageName = self.before_imagepath.split('/')[-1].split('.')[0]
            scaled_pixmap = pixmap.scaled(600, int(600 / self.image_w * self.image_h))
            self.imageLabel1.setPixmap(scaled_pixmap)

    def readFile(self):
        # 从文件中读取内容并显示在 QTextEdit 控件中
        try:
            with open(self.txt_path, 'r') as file:
                self.textEdit.setText(file.read())
        except FileNotFoundError:
            self.textEdit.setText("File not found.")


    def makeFiledir(self):
        """ 创建输出文件夹"""
        if not os.path.exists(self.outputPath):  # 是否存在这个文件夹
            os.makedirs(self.outputPath)  # 如果没有这个文件夹，那就创建一个

    def perpendicular_bisector(self, p1, p2):
        """计算两点间的垂直平分线"""
        midpoint = (p1 + p2) / 2
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        perp_slope = -1 / slope if slope != 0 else float('inf')

        # 垂直平分线的方程 y = mx + c
        if perp_slope == float('inf'):
            return None, midpoint[0]  # 垂直线 x = constant
        else:
            c = midpoint[1] - perp_slope * midpoint[0]
            return perp_slope, c

    def intersection(self, line1, line2):
        """计算两条线的交点"""
        m1, c1 = line1
        m2, c2 = line2

        if m1 is None and m2 is not None:
            x = c1
            y = m2 * x + c2
        elif m2 is None and m1 is not None:
            x = c2
            y = m1 * x + c1
        elif m1 == m2:
            return None  # 平行线无交点
        else:
            x = (c2 - c1) / (m1 - m2)
            y = m1 * x + c1

        return x, y

    def Drawpic(self, points, img_path):
        # 存储所有的垂直平分线
        bisectors = []
        num_points = len(points)
        for i in range(num_points):
            for j in range(i + 1, num_points):
                bisector = self.perpendicular_bisector(points[i], points[j])
                bisectors.append(bisector)
        print(len(bisectors))

        # 找出所有垂直平分线的交点
        intersections = []
        for i in range(len(bisectors)):
            for j in range(i + 1, len(bisectors)):
                intersect = self.intersection(bisectors[i], bisectors[j])
                if intersect is not None:
                    intersections.append(intersect)
        # print(intersections)
        print(len(intersections))
        # # 计算所有交点坐标的平均值
        # if intersections:
        #     avg_x = sum(point[0] for point in intersections) / len(intersections)
        #     avg_y = sum(point[1] for point in intersections) / len(intersections)
        #     average_point = (avg_x, avg_y)
        # else:
        #     average_point = None
        # 计算所有交点坐标的平均值，自动剔除离群交点
        if intersections:
            intersections_np = np.array(intersections)
            # 初步中心点
            rough_center = intersections_np.mean(axis=0)
            # 计算每个点到中心点的欧氏距离
            distances = np.linalg.norm(intersections_np - rough_center, axis=1)
            # # 使用距离的均值和标准差来过滤离群点
            # dist_mean = np.mean(distances)
            # dist_std = np.std(distances)
            # threshold = dist_mean + 0.5 * dist_std  # 可调整系数，例如 2.0 更严格
            # 使用中位数与 IQR方法（对非正态分布更稳健）
            median = np.median(distances)
            iqr = np.percentile(distances, 75) - np.percentile(distances, 25)
            threshold = median + 0.5 * iqr
            # 过滤距离超过阈值的点
            inliers = intersections_np[distances <= threshold]

            if len(inliers) > 0:
                avg_x, avg_y = np.mean(inliers, axis=0)
                average_point = (avg_x, avg_y)
            else:
                average_point = None
        else:
            average_point = None

        print(f"保留交点数: {len(inliers)} / {len(intersections)}")
        print("Average Intersection Point:", average_point)

        # 复制图像避免修改原图
        img = cv2.imread(img_path)
        # 图像大小
        height, width = img.shape[:2]

        # 1. 画点（蓝色圆圈）
        for x, y in points.astype(int):
            cv2.circle(img, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)  # BGR：蓝色


        # # 3. 绘制所有垂直平分线
        # for m, c in bisectors:
        #     if m is None:  # 垂直线 x = c
        #         x = int(c)
        #         cv2.line(img, (x, 0), (x, height), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        #     else:
        #         # 计算 y = mx + c 在图像左右边界的两个点
        #         x1, x2 = 0, width
        #         y1 = int(m * x1 + c)
        #         y2 = int(m * x2 + c)
        #         if 0 <= y1 <= height or 0 <= y2 <= height:  # 至少有部分线段在图像内
        #             cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        # 3. 绘制所有垂直平分线（限制在图像内）
        for m, c in bisectors:
            points_to_draw = []
            if m is None:  # 垂直线 x = c
                x = int(c)
                if 0 <= x < width:
                    points_to_draw = [(x, 0), (x, height - 1)]
            else:
                # 计算与图像上下左右边界的交点
                # 与左边界 (x = 0)
                y = m * 0 + c
                if 0 <= y < height:
                    points_to_draw.append((0, int(y)))
                # 与右边界 (x = width-1)
                y = m * (width - 1) + c
                if 0 <= y < height:
                    points_to_draw.append((width - 1, int(y)))
                # 与上边界 (y = 0)
                if m != 0:
                    x = -c / m
                    if 0 <= x < width:
                        points_to_draw.append((int(x), 0))
                # 与下边界 (y = height-1)
                if m != 0:
                    x = (height - 1 - c) / m
                    if 0 <= x < width:
                        points_to_draw.append((int(x), height - 1))

            # 如果找到两个图像内的边界交点，画线
            if len(points_to_draw) >= 2:
                pt1, pt2 = points_to_draw[:2]
                cv2.line(img, pt1, pt2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        # 2. 画平均交点（红色叉号）
        if average_point is not None:
            x, y = map(int, average_point)
            cv2.line(img, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), 2)  # 红色
            cv2.line(img, (x - 6, y + 6), (x + 6, y - 6), (0, 0, 255), 2)

        # # 211. 画保留的交点（红色叉号）
        # if inliers is not None:
        #     for (x, y) in inliers:
        #         x, y = int(x), int(y)
        #         cv2.line(img, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), 2)
        #         cv2.line(img, (x - 6, y + 6), (x + 6, y - 6), (0, 0, 255), 2)

        # 4. 保存图像
        cv2.imwrite(self.outputPath + self.imageName + "perpendicular_bisectors_cv2.png", img)


    def Readvalue(self):
        try:
            # 检测指针、圆心
            self.yolo1 = YOLO(model=self.model_path1, task='detect')
            self.result1 = self.yolo1(source=self.before_imagepath, save=True, conf=0.7)
            # 检测刻度
            self.yolo2 = YOLO(model=self.model_path2, task='detect')
            self.result2 = self.yolo2(source=self.before_imagepath, save=True)

            # x,y图像中心点
            image_h, image_w = self.result1[0].orig_shape
            y = image_h / 2
            x = image_w / 2
            # print(x, y)
            # print(image_h, image_w)

            data_p = self.result1[0].keypoints.xy.cpu().numpy()
            # s1 s2 为指针端点， s3 s4为圆心
            s1, s2 = data_p[0][0][0], data_p[0][0][1]
            s3, s4 = data_p[0][1][0], data_p[0][1][1]
            distance1 = math.sqrt((s1 - x) ** 2 + (s2 - y) ** 2)
            distance2 = math.sqrt((s3 - x) ** 2 + (s4 - y) ** 2)
            # print(distance1, distance2)
            if distance1 < distance2:
                # 交换(s1, s2)和(s3, s4)的位置
                s1, s2, s3, s4 = s3, s4, s1, s2
            print(s1, s2, s3, s4)

            # 0刻度：k0_0, k0_1  结束25刻度：k25_0, k25_1
            data_k = self.result2[0].keypoints.xy.cpu().numpy()
            k1, k2 = data_k[0][0]
            k3, k4 = data_k[1][0]
            k5, k6 = data_k[2][0]
            k7, k8 = data_k[3][0]
            k9, k10 = data_k[4][0]
            k11, k12 = data_k[5][0]
            # print(k1, k2)
            # print(k3, k4)
            # print(k5, k6)
            # print(k7, k8)
            # print(k9, k10)
            # print(k11, k12)

            # 定义六点的坐标
            points = np.array([
                [k1, k2],
                [k3, k4],
                [k5, k6],
                [k7, k8],
                [k9, k10],
                [k11, k12]
            ])

            # 计算各个刻度与图像左下角距离 距离最短的即为0刻度坐标
            d_k12 = math.sqrt((k1 - 0) ** 2 + (k2 - image_h) ** 2)
            d_k34 = math.sqrt((k3 - 0) ** 2 + (k4 - image_h) ** 2)
            d_k56 = math.sqrt((k5 - 0) ** 2 + (k6 - image_h) ** 2)
            d_k78 = math.sqrt((k7 - 0) ** 2 + (k8 - image_h) ** 2)
            d_k910 = math.sqrt((k9 - 0) ** 2 + (k10 - image_h) ** 2)
            d_k1112 = math.sqrt((k11 - 0) ** 2 + (k12 - image_h) ** 2)
            d_min1 = min(d_k12, d_k34, d_k56, d_k78, d_k910, d_k1112)
            # print(d_k12, d_k34, d_k56, d_k78, d_k910, d_k1112)
            # print(d_min1)
            if d_min1 == d_k12:
                k0_0, k0_1 = k1, k2
            elif d_min1 == d_k34:
                k0_0, k0_1 = k3, k4
            elif d_min1 == d_k56:
                k0_0, k0_1 = k5, k6
            elif d_min1 == d_k78:
                k0_0, k0_1 = k7, k8
            elif d_min1 == d_k910:
                k0_0, k0_1 = k9, k10
            elif d_min1 == d_k1112:
                k0_0, k0_1 = k11, k12

            # 计算各个刻度与图像右下角距离 距离最短的即为终止刻度坐标
            dd_k12 = math.sqrt((k1 - image_w) ** 2 + (k2 - image_h) ** 2)
            dd_k34 = math.sqrt((k3 - image_w) ** 2 + (k4 - image_h) ** 2)
            dd_k56 = math.sqrt((k5 - image_w) ** 2 + (k6 - image_h) ** 2)
            dd_k78 = math.sqrt((k7 - image_w) ** 2 + (k8 - image_h) ** 2)
            dd_k910 = math.sqrt((k9 - image_w) ** 2 + (k10 - image_h) ** 2)
            dd_k1112 = math.sqrt((k11 - image_w) ** 2 + (k12 - image_h) ** 2)
            d_min2 = min(dd_k12, dd_k34, dd_k56, dd_k78, dd_k910, dd_k1112)
            # print(dd_k12, dd_k34, dd_k56, dd_k78, dd_k910, dd_k1112)
            # print(d_min2)

            if d_min2 == dd_k12:
                k25_0, k25_1 = k1, k2
            elif d_min2 == dd_k34:
                k25_0, k25_1 = k3, k4
            elif d_min2 == dd_k56:
                k25_0, k25_1 = k5, k6
            elif d_min2 == dd_k78:
                k25_0, k25_1 = k7, k8
            elif d_min2 == dd_k910:
                k25_0, k25_1 = k9, k10
            elif d_min2 == dd_k1112:
                k25_0, k25_1 = k11, k12

            # print(k0_0, k0_1)
            # print(k25_0, k25_1)

            # 计算0刻度、圆心、指针端点所成角度
            v1 = [k0_0 - s3, k0_1 - s4]
            v2 = [s1 - s3, s2 - s4]
            theta = self.GetClockAngle(v1, v2)
            t1 = round(theta, 2)

            # 计算0刻度、圆心、终止刻度所成角度
            v3 = [k0_0 - s3, k0_1 - s4]
            v4 = [k25_0 - s3, k25_1 - s4]
            theta2 = self.GetClockAngle(v3, v4)
            t2 = round(theta2, 2)

            # 由于表盘0刻度少一格做补偿
            division = 25 / theta2
            # readValue = self.divisionValue * theta
            readValue = division * theta + 0.5
            r1 = round(readValue, 2)

            self.result_string = str(r1)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.txt_label1.setText(current_time)
            self.txt_label2.setText('The reading of the instrument is: ' + self.result_string)
            self.txt_label4.setText('0刻度和检测指针所成角度为: ' + str(t1))
            self.txt_label5.setText('0刻度和结束刻度所成角度为: ' + str(t2))
            with open(self.txt_path, 'a') as file:
                file.write(current_time)
                file.write("\n0刻度和检测指针所成角度为: " + str(t1))
                file.write("\n0刻度和结束刻度所成角度为: " + str(t2))
                file.write("\nThe reading of the instrument is: " + self.result_string + '\n')

            img_path = self.flie_path + self.result2[0].save_dir + '/' + self.result2[0].path.split('\\')[-1]
            pixmap = QPixmap(img_path)
            # self.imageLabel1.setPixmap(pixmap.scaled(self.imageLabel1.size(), Qt.KeepAspectRatio))
            scaled_pixmap = pixmap.scaled(600, int(600 / self.image_w * self.image_h))  # 将图像缩放
            self.imageLabel2.setPixmap(scaled_pixmap)
            self.Drawpic(points, img_path)
        except IndexError as e:  # 未捕获到异常，程序直接报错
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.txt_label1.setText(current_time)
            self.txt_label2.setText('The reading of the instrument is: ')
            self.txt_label4.setText('未检测到读数')
            self.txt_label5.setText('未检测到读数')
            with open(self.txt_path, 'a') as file:
                file.write(current_time)
                file.write("\n未检测到读数\n")


# 创建应用程序和窗口实例，并运行应用程序
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageDetection()
    ex.show()
    sys.exit(app.exec_())
