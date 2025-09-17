# image_hsv_init.py
import sys
import os

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from image_find_focus import FocusFinder

import cv2

# 初始化 FocusFinder
focus_finder = FocusFinder()

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取一帧
    ret, frame = cap.read()
    
    if not ret:
        print("无法读取帧")
        break

    cv2.imshow('frame Image', frame)

    # 对图像进行处理
    focus_image, has_res = focus_finder.find_focus(frame)
    
    # 显示处理后的图像
    cv2.imshow('Focus Image', focus_image)
    
    # 按下 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()