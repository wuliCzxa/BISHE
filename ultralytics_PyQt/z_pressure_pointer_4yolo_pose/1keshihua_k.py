import os
import cv2
import numpy as np
import json

import matplotlib.pyplot as plt

""""注意：把图片与标签（json）放在同一个文件夹里 """
"""使用：  只需要修改files_path路径 与 kpt_color_map就行。 kpt_color_map是关键点类别的配置，我的是三个关键点类别，分别叫angle_30, angle_60, angle_90
          你把这里改成自己的类别即可，如果类别小于3个就减少配置，如果大于3个就按照格式增加配置 """

num = 0
# 图片与标签json的文件夹路径
files_path = r"mydata_visualization_k"

for file in os.listdir(files_path):
    if file.endswith(".jpg"):
        img_path = os.path.join(files_path, file)
        img_bgr = cv2.imread(img_path)

        file_profix = os.path.splitext(file)[0]
        labelme_name = file_profix + '.json'  # DSC_0219.json
        labelme_path = os.path.join(files_path, labelme_name)
        # 载入labelme格式的json标注文件
        with open(labelme_path, 'r', encoding='utf-8') as f:
            labelme = json.load(f)

        # <<<<<<<<<<<<<<<<<<可视化框（rectangle）标注>>>>>>>>>>>>>>>>>>>>>
        # 框可视化配置
        bbox_color = (255, 129, 0)  # 框的颜色
        bbox_thickness = 5  # 框的线宽

        # 框类别文字
        bbox_labelstr = {
            'font_size': 1,  # 字体大小
            'font_thickness': 2,  # 字体粗细
            'offset_x': 0,  # X 方向，文字偏移距离，向右为正
            'offset_y': 0,  # Y 方向，文字偏移距离，向下为正
        }
        # 画框
        for each_ann in labelme['shapes']:  # 遍历每一个标注
            if each_ann['shape_type'] == 'rectangle':  # 筛选出框标注
                # 框的类别
                bbox_label = each_ann['label']
                # 框的两点坐标
                bbox_keypoints = each_ann['points']
                bbox_keypoint_A_xy = bbox_keypoints[0]
                bbox_keypoint_B_xy = bbox_keypoints[1]
                # 左上角坐标
                bbox_top_left_x = int(min(bbox_keypoint_A_xy[0], bbox_keypoint_B_xy[0]))
                bbox_top_left_y = int(min(bbox_keypoint_A_xy[1], bbox_keypoint_B_xy[1]))
                # 右下角坐标
                bbox_bottom_right_x = int(max(bbox_keypoint_A_xy[0], bbox_keypoint_B_xy[0]))
                bbox_bottom_right_y = int(max(bbox_keypoint_A_xy[1], bbox_keypoint_B_xy[1]))

                # 画矩形：画框
                img_bgr = cv2.rectangle(img_bgr, (bbox_top_left_x, bbox_top_left_y),
                                        (bbox_bottom_right_x, bbox_bottom_right_y),
                                        bbox_color, bbox_thickness)
                # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
                img_bgr = cv2.putText(img_bgr, bbox_label, (
                    bbox_top_left_x + bbox_labelstr['offset_x'],
                    bbox_top_left_y + bbox_labelstr['offset_y']),
                                      cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                                      bbox_labelstr['font_thickness'])
        # # 可视化
        # plt.imshow(img_bgr[:, :, ::-1])  # 将bgr通道转换成rgb通道
        # plt.show()

        # <<<<<<<<<<<<<<<<<<可视化关键点（keypoint）标注>>>>>>>>>>>>>>>>>>>>>
        # 关键点的可视化配置
        # 关键点配色
        kpt_color_map = {
            'k1': {'id': 0, 'color': [255, 0, 0], 'radius': 3, 'thickness': -1},
            # '2': {'id': 1, 'color': [0, 255, 0], 'radius': 3, 'thickness': -1}
        }

        # 点类别文字
        kpt_labelstr = {
            'font_size': 1,  # 字体大小
            'font_thickness': 2,  # 字体粗细
            'offset_x': 0,  # X 方向，文字偏移距离，向右为正
            'offset_y': 0,  # Y 方向，文字偏移距离，向下为正
        }

        # 画点
        for each_ann in labelme['shapes']:  # 遍历每一个标注
            if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                kpt_label = each_ann['label']  # 该点的类别
                # 该点的 XY 坐标
                kpt_xy = each_ann['points'][0]
                kpt_x, kpt_y = int(kpt_xy[0]), int(kpt_xy[1])

                # 该点的可视化配置
                kpt_color = kpt_color_map[kpt_label]['color']  # 颜色
                kpt_radius = kpt_color_map[kpt_label]['radius']  # 半径
                kpt_thickness = kpt_color_map[kpt_label]['thickness']  # 线宽（-1代表填充）
                # 画圆：画该关键点
                img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, kpt_thickness)
                # 写该点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
                img_bgr = cv2.putText(img_bgr, kpt_label,
                                      (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                                      cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                                      kpt_labelstr['font_thickness'])

        # 可视化
        # plt.imshow(img_bgr[:, :, ::-1])  # 将bgr通道转换成rgb通道
        # plt.show()
        # 当前目录下保存可视化结果
        cv2.imwrite(f'visualization_result_k/{num}.jpg', img_bgr)
        num += 1
print('=========================ok========================')