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
        self.flie_path = 'D:/毕设/ultralytics/z_pressure_pointer_3yolo/'  # z_pressure_pointer_3yolo绝对路径
        self.model_path = 'weight/weights/best.pt'    # 权重文件路径
        self.txt_path = self.flie_path + 'Result_pressure_pointer_3yolo.txt'  # 读数保存路径
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

    def Readvalue(self):
        try:
            self.yolo = YOLO(model=self.model_path, task='detect')
            self.result = self.yolo(source=self.before_imagepath, save=True, conf=0.7)

            data = self.result[0].boxes.data.cpu().numpy()

            # 使用argsort()方法对第一列进行排序，并获取排序后的索引
            sorted_indices = np.argsort(data[:, -1])
            data_sorted = data[sorted_indices]

            # 0刻度线坐标
            x_0 = data_sorted[0][0]
            y_0 = data_sorted[0][3]
            # 终止刻度线坐标
            x_25 = data_sorted[1][2]
            y_25 = data_sorted[1][3]
            # 指针端点坐标
            x_p = data_sorted[2][0]
            y_p = data_sorted[2][1]
            # 圆心坐标
            x_c = (data_sorted[3][0] + data_sorted[3][2]) / 2
            y_c = (data_sorted[3][1] + data_sorted[3][3]) / 2

            image = cv2.imread(self.before_imagepath)
            point1 = (int(x_0), int(y_0))
            point2 = (int(x_25), int(y_25))
            point3 = (int(x_p), int(y_p))
            point4 = (int(x_c), int(y_c))

            cv2.circle(image, point1, 5, (0, 0, 255), -1)
            cv2.circle(image, point2, 5, (0, 0, 255), -1)
            cv2.circle(image, point3, 5, (0, 0, 255), -1)
            cv2.circle(image, point4, 5, (0, 0, 255), -1)
            cv2.imwrite(self.outputPath + self.imageName + "fitting.jpg", image)


            v1 = [x_0 - x_c, y_0 - y_c]
            v2 = [x_p - x_c, y_p - y_c]
            theta = self.GetClockAngle(v1, v2)
            t1 = round(theta, 2)

            v3 = [x_0 - x_c, y_0 - y_c]
            v4 = [x_25 - x_c, y_25 - y_c]

            theta2 = self.GetClockAngle(v3, v4)
            t2 = round(theta2, 2)

            # 由于表盘0刻度少一格做补偿
            division = 25 / theta2
            # readValue = self.divisionValue * theta
            readValue = division * theta + 0.3
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
