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

class Functions:
    @staticmethod
    def GetClockAngle(v1, v2):
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

    @staticmethod
    def Distances(a, b):
        # 返回两点间的距离
        x1, y1 = a
        x2, y2 = b
        Distances = int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        return Distances

    @staticmethod
    def couputeMean(deg):
        # 对数据进行处理，提取均值
        if (True):
            # new_nums = list(set(deg)) #剔除重复元素
            mean = np.mean(deg)
            var = np.var(deg)
            percentile = np.percentile(deg, (25, 50, 75), method='midpoint')
            # 以下为箱线图的五个特征值
            Q1 = percentile[0]  # 上四分位数
            Q3 = percentile[2]  # 下四分位数
            IQR = Q3 - Q1  # 四分位距
            ulim = Q3 + 2.5 * IQR  # 上限 非异常范围内的最大值
            llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值

            new_deg = []
            uplim = []
            for i in range(len(deg)):
                if (llim < deg[i] and deg[i] < ulim):
                    new_deg.append(deg[i])
        new_deg = np.mean(new_deg)

        return new_deg

class ImageDetection(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.before_imagepath = None
        self.later_imagepath = None
        self.result_string = None
        self.flie_path = 'C:/Users/xxa/Desktop/ultralytics/z_pressure_pointer_1opencv_yolo/'  # ultralytics存放路径
        self.model_path = 'weight/weights/best.pt'    # 权重文件路径
        # self.model_path = 'runs/best.pt'    # 权重文件路径
        self.txt_path = self.flie_path +'Result_pressure_pointer_1opencv_yolo.txt'  # 读数保存路径
        self.imagepath = self.before_imagepath
        self.image = None
        self.outputPath = ('outputs/')
        self.imageName = None
        self.circleimg = None
        self.panMask = None  # 霍夫圆检测切割的表盘图片
        self.poniterMask = None  # 指针图片
        self.numLineMask = None  # 刻度线图片
        self.centerPoint = None  # 中心点[x,y]
        self.farPoint = None  # 指针端点[x,y]
        self.zeroPoint = None  # 起始点[x,y]
        self.endPoint = None
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
        self.topLabel = QLabel('指针式仪表读数检测识别系统')
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
        self.txt_label1.setFont(font)   #时间
        self.txt_label2.setFont(font)   #结果1
        self.txt_label4.setFont(font)   #结果2
        self.txt_label5.setFont(font)   #结果3
        self.imageLabel1.setFont(font)
        self.imageLabel2.setFont(font)
        self.txt_label3.setFont(font2)  #检测结果
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

    def ImgCutCircle(self):
        # 截取表盘区域，滤除背景
        img = self.image
        dst = cv2.pyrMeanShiftFiltering(img, 10, 100)  # 图像在色彩层面的平滑滤波，它可以中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域，第三个参数sp，定义的漂移物理空间半径大小；第四个参数sr，定义的漂移色彩空间半径大小；
        # cv2.imshow('dst', dst)
        cv2.imwrite(self.outputPath + self.imageName + '_dst.jpg', dst)
        cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.outputPath + self.imageName + '_cvt.jpg', cimage)
        # cv2.imshow('cimage', cimage)
        circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=80, maxRadius=0)
        cv2.imwrite(self.outputPath + self.imageName + '_circles.jpg', circles)
        circles = np.uint16(np.around(circles))  # 把类型换成整数
        r_1 = circles[0, 0, 2]
        c_x = circles[0, 0, 0]
        c_y = circles[0, 0, 1]
        circle = np.ones(img.shape, dtype="uint8")
        circle = circle * 255
        cv2.circle(circle, (c_x, c_y), int(r_1), 0, -1)
        bitwiseOr = cv2.bitwise_or(img, circle)
        cv2.imwrite(self.outputPath + self.imageName + '_1_imgCutCircle.jpg', bitwiseOr)
        self.cirleData = [r_1, c_x, c_y]
        self.panMask = bitwiseOr

        return bitwiseOr

    def ContoursFilter(self):
        # 对轮廓进行筛选
        r_1, c_x, c_y = self.cirleData

        img = self.panMask.copy()
        img = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imwrite(self.outputPath + self.imageName + '_gaosilvbo.jpg', img)
        # img = cv2.medianBlur(img, 5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.outputPath + self.imageName + '_huidutu.jpg', gray)
        binary = cv2.adaptiveThreshold(~gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
        cv2.imwrite(self.outputPath + self.imageName + '_binary.jpg', binary)
        # 轮廓查找，根据版本不同，返回参数不同
        if cv2.__version__ > '4.0.0':
            contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            aa, contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cntset = []  # 刻度线轮廓集合
        cntareas = []  # 刻度线面积集合

        needlecnt = []  # 指针轮廓集合
        needleareas = []  # 指针面积集合
        radiusLength = [r_1 * 0.6, r_1 * 1]  # 半径范围

        localtion = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            # （中心点坐标，（宽度，高度）,旋转的角度）=   = rect
            a, (w, h), c = rect
            w = int(w)
            h = int(h)
            if h == 0 or w == 0:
                pass
            else:
                dis = Functions.Distances((c_x, c_y), a)
                # if (radiusLength[0] < dis and radiusLength[e] > dis):
                if (radiusLength[0] < dis and radiusLength[1] > dis):
                    # 矩形筛选
                    if h / w > 4 or w / h > 4:
                        localtion.append(dis)
                        cntset.append(cnt)
                        cntareas.append(w * h)
                else:
                    if w > r_1 / 2 or h > r_1 / 2:
                        needlecnt.append(cnt)
                        needleareas.append(w * h)
        cntareas = np.array(cntareas)
        areasMean = Functions.couputeMean(cntareas)  # 中位数，上限区
        new_cntset = []
        # 面积
        for i, cnt in enumerate(cntset):
            if (cntareas[i] <= areasMean * 1.5 and cntareas[i] >= areasMean * 0.8):
                new_cntset.append(cnt)

        self.r = np.mean(localtion)
        mask = np.zeros(img.shape[0:2], np.uint8)
        self.poniterMask = cv2.drawContours(mask, needlecnt, -1, (255, 255, 255), -1)  # 生成掩膜

        mask = np.zeros(img.shape[0:2], np.uint8)
        self.numLineMask = cv2.drawContours(mask, new_cntset, -1, (255, 255, 255), -1)  # 生成掩膜
        cv2.imwrite(self.outputPath + self.imageName + '_2_numLineMask.jpg', self.numLineMask)
        cv2.imwrite(self.outputPath + self.imageName + '_3_poniterMask.jpg', self.poniterMask)

        self.new_cntset = new_cntset

        return new_cntset

    def FitNumLine(self):
        """ 轮廓拟合直线"""
        lineSet = []  # 拟合线集合
        img = self.image.copy()
        for cnt in self.new_cntset:
            rect = cv2.minAreaRect(cnt)
            # 获取矩形四个顶点，浮点型
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.polylines(img, [box], True, (0, 255, 0), 1)  # pic
            output = cv2.fitLine(cnt, 2, 0, 0.001, 0.001)
            k = output[1] / output[0]
            k = round(k[0], 2)
            b = output[3] - k * output[2]
            b = round(b[0], 2)
            x1 = 1
            x2 = img.shape[0]
            y1 = int(k * x1 + b)
            y2 = int(k * x2 + b)
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), e)
            # lineSet:刻度线拟合直线数组，k斜率 b
            lineSet.append([k, b])  # 求中心点的点集[k,b]
        cv2.imwrite(self.outputPath + self.imageName + '_4_fitNumLine.jpg', img)
        self.lineSet = lineSet
        return lineSet

    def getIntersectionPoints(self):
        # 获取刻度线交点
        img = cv2.imread(self.before_imagepath)
        lineSet = self.lineSet
        h, w, c = img.shape  # 注意是 h, w, c（OpenCV 是行高在前）
        xlist = []
        ylist = []

        if len(lineSet) > 2:
            lkb = int(len(lineSet) / 2)
            kb1 = lineSet[0:lkb]
            kb2 = lineSet[lkb:(2 * lkb)]
        else:
            # 若只有两条线，则直接组合
            kb1 = [lineSet[0]]
            kb2 = [lineSet[1]]

        for wx in kb1:
            for wy in kb2:
                k1, b1 = wx
                k2, b2 = wy
                try:
                    # 防止除0或斜率相等
                    if (k1 - k2) == 0:
                        k1 += 1e-5
                    x = (b2 - b1) / (k1 - k2)
                    y = k1 * x + b1
                    x = int(round(x))
                    y = int(round(y))
                except:
                    # 兜底处理
                    x = int(round((b2 - b1 - 0.01) / (k1 - k2 + 0.01)))
                    y = int(round(k1 * x + b1))

                if 0 <= x <= w and 0 <= y <= h:
                    xlist.append(x)
                    ylist.append(y)

        if not xlist or not ylist:
            self.centerPoint = [0, 0]
            return img

        cx = int(np.mean(xlist))
        cy = int(np.mean(ylist))
        self.centerPoint = [cx, cy]
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
        cv2.imwrite(self.outputPath + self.imageName + '_5_IntersectionPoints.jpg', img)
        return img

    def getIntersectionPoints2(self):
        # 获取刻度线交点
        img = cv2.imread(self.before_imagepath)
        lineSet = self.lineSet
        w, h, c = img.shape
        point_list = []
        xlist = []
        ylist = []
        if len(lineSet) > 2:
            np.random.shuffle(lineSet)
            lkb = int(len(lineSet) / 2)
            kb1 = lineSet[0:lkb]
            kb2 = lineSet[lkb:(2 * lkb)]
            kb1sample = random.sample(kb1, int(len(kb1) / 2))
            kb2sample = random.sample(kb2, int(len(kb2) / 2))
        else:
            kb1sample = lineSet[0]
            kb2sample = lineSet[1]
        for i, wx in enumerate(kb1sample):
            # for wy in kb2:
            for wy in kb2sample:
                k1, b1 = wx
                k2, b2 = wy
                # k1-->[123]
                try:
                    if (b2 - b1) == 0:
                        b2 = b2 - 0.1
                    if (k1 - k2) == 0:
                        k1 = k1 - 0.1
                    x = (b2 - b1) / (k1 - k2)
                    y = k1 * x + b1
                    x = int(round(x))
                    y = int(round(y))
                except:
                    x = (b2 - b1 - 0.01) / (k1 - k2 + 0.01)
                    y = k1 * x + b1
                    x = int(round(x))
                    y = int(round(y))
                # x,y=solve_point(k1, b1, k2, b2)
                if x < 0 or y < 0 or x > w or y > h:
                    break
                xlist.append(x)
                ylist.append(y)

        cx = int(np.mean(xlist))
        cy = int(np.mean(ylist))
        self.centerPoint = [cx, cy]
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
        cv2.imwrite(self.outputPath + self.imageName + '_5_IntersectionPoints.jpg', img)
        return img

    def FitPointerLine(self):
        # 拟合指针直线段
        img = self.poniterMask
        orgin_img = self.image.copy()
        lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=int(self.r / 2), maxLineGap=2)
        dmax = 0
        pointerLine = []
        # 最长的线段为指针
        for line in lines:
            x1, y1, x2, y2 = line[0]
            d1 = Functions.Distances((x1, y1), (x2, y2))
            if d1 > dmax:
                dmax = d1
                pointerLine = line[0]
        x1, y1, x2, y2 = pointerLine
        d1 = Functions.Distances((x1, y1), (self.centerPoint[0], self.centerPoint[1]))
        d2 = Functions.Distances((x2, y2), (self.centerPoint[0], self.centerPoint[1]))
        if d1 > d2:
            self.farPoint = [x1, y1]
        else:
            self.farPoint = [x2, y2]

        cv2.line(orgin_img, (x1, y1), (x2, y2), 20, 1, cv2.LINE_AA)
        cv2.circle(orgin_img, (self.farPoint[0], self.farPoint[1]), 2, (0, 0, 255), 2)
        cv2.imwrite(self.outputPath + self.imageName + '_6_PointerLine.jpg', img)
        cv2.imwrite(self.outputPath + self.imageName + '_7_PointerPoint.jpg', orgin_img)

    def Readvalue(self):
        self.yolo = YOLO(model=self.model_path, task='detect')
        self.result = self.yolo(source=self.before_imagepath, save=True, conf=0.6)

        data = self.result[0].boxes.data.cpu().numpy()
        # 使用argsort()方法对第一列进行排序，并获取排序后的索引
        sorted_indices = np.argsort(data[:, 0])
        data_sorted = data[sorted_indices]
        self.zeroPoint = [float(data_sorted[0][0]), float(data_sorted[0][3])]
        self.endPoint = [float(data_sorted[1][2]), float(data_sorted[1][3])]

        self.ImgCutCircle()
        self.ContoursFilter()
        self.FitNumLine()
        self.getIntersectionPoints()
        self.FitPointerLine()

        v1 = [self.zeroPoint[0] - self.centerPoint[0], self.zeroPoint[1] - self.centerPoint[1]]
        v2 = [self.farPoint[0] - self.centerPoint[0], self.farPoint[1] - self.centerPoint[1]]
        theta = Functions.GetClockAngle(v1, v2)
        t1 = round(theta, 2)

        v3 = [self.zeroPoint[0] - self.centerPoint[0], self.zeroPoint[1] - self.centerPoint[1]]
        v4 = [self.endPoint[0] - self.centerPoint[0], self.endPoint[1] - self.centerPoint[1]]

        theta2 = Functions.GetClockAngle(v3, v4)
        t2 = round(theta2, 2)
        division = 26 / theta2
        readValue = division * theta
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


        img_path = self.flie_path + self.result[0].save_dir + '/' + self.result[0].path.split('\\')[-1]
        pixmap = QPixmap(img_path)
        # self.imageLabel1.setPixmap(pixmap.scaled(self.imageLabel1.size(), Qt.KeepAspectRatio))
        scaled_pixmap = pixmap.scaled(600, int(600 / self.image_w * self.image_h))  # 将图像缩放
        self.imageLabel2.setPixmap(scaled_pixmap)

# 创建应用程序和窗口实例，并运行应用程序
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageDetection()
    ex.show()
    sys.exit(app.exec_())
