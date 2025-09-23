# -*- coding: utf-8 -*-
import time
import cv2
import threading
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import Global_variables
import AB_optimize
from ql_main import ql_main
from yolov5.detect_self import YoloDetecter
from image_find_focus import FocusFinder
from Tools import coordinate_mapping, coordinate_to_pos, pos_to_coordinate, Check, get_video_frame, safe_detect
import json
import socket
# 导入Qt界面
from qt_YOLO import MainWindow
from PyQt5.QtWidgets import QApplication
detect_flag = True
# 玩家历史落子
history_set = set()

# AI上一步落子位置
ai_down_last = (None, None)
# UDP客户端用于发送YOLO数据到Qt界面
YOLO_UDP_IP = "127.0.0.1"
YOLO_UDP_PORT = 5005
yolo_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

WIDTH_GOBANG = Global_variables.WIDTH_GOBANG  # 五子棋盘总宽度
LENGTH_GOBANG = Global_variables.LENGTH_GOBANG  # 五子棋盘总长度
WIDTH_ERR_GOBANG = Global_variables.WIDTH_ERR_GOBANG  # 五子棋盘内外边框间距(宽度方向)
LENGTH_ERR_GOBANG = Global_variables.LENGTH_ERR_GOBANG  # 五子棋盘内外边框间距(长度方向)
ROW_GOBANG = Global_variables.ROW_GOBANG  # 五子棋盘行数(宽度方向)
COLUMN_GOBANG = Global_variables.COLUMN_GOBANG  # 五子棋盘列数(长度方向)


def find_last_down_pos(now_pos_set):
    global history_set
    our_down_pos = now_pos_set - history_set
    if len(our_down_pos) == 0:
        print("该你下了哦！")
        return None
    if len(our_down_pos) > 1:
        print("你下了%d个棋子，不许耍赖哦!" % (len(our_down_pos)))
        return None
    our_down_pos_x, our_down_pos_y = list(our_down_pos)[0]
    if our_down_pos_x >= ROW_GOBANG or our_down_pos_y >= COLUMN_GOBANG:
        print("下错了！请下在棋盘范围内！")
        return None
    history_set = now_pos_set
    return our_down_pos_x, our_down_pos_y


def cv2_to_Img(img):
    if img is not None:
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        Img = Image.fromarray(cv2image)
    return Img


def get_root():
    file = Path(__file__).resolve()
    parent_dir = file.parent
    root = str(parent_dir).replace("\\", "/")
    return root


def yolo_to_pixel(yolo_list, rows_b, cols_b):
    data = []
    for x, y, w, h, conf, c in yolo_list:
        pixel_y = y * cols_b
        pixel_x = x * rows_b
        data.append([pixel_x, pixel_y, conf, c])
    return data


def coordinate_to_pos(coordinate_list, go_stones):
    pos_set = set()
    ai_pos_set = set()
    pos_set_conf = []
    ai_pos_set_conf = []
    player_class = 1 if go_stones == "white" else 0

    for coordinate_x, coordinate_y, conf, c in coordinate_list:
        pos_x = round(abs(coordinate_x - WIDTH_ERR_GOBANG) / (WIDTH_GOBANG - 2 * WIDTH_ERR_GOBANG) * (ROW_GOBANG - 1))
        pos_y = round(
            abs(coordinate_y - LENGTH_ERR_GOBANG) / (LENGTH_GOBANG - 2 * LENGTH_ERR_GOBANG) * (COLUMN_GOBANG - 1))
        if c == player_class:
            pos_set.add((pos_x, pos_y))
            pos_set_conf.append((pos_x, pos_y, float(conf)))
        else:
            ai_pos_set.add((pos_x, pos_y))
            ai_pos_set_conf.append((pos_x, pos_y, float(conf)))
    return pos_set, ai_pos_set, pos_set_conf, ai_pos_set_conf


def coordinate_mapping(pixel_list, physical_rows, physical_cols, pixel_rows, pixel_cols):
    data = []
    for x, y, conf, c in pixel_list:
        x = x * physical_rows / pixel_rows
        y = y * physical_cols / pixel_cols
        data.append([x, y, conf, c])
    return data


def detct(image, self_yolo, mod, go_stones, status_now):
    global ai_down_last
    # 图像校正和目标区域提取
    focus_finder = FocusFinder()
    focus_image, has_res = focus_finder.find_focus(image)
    cv2.imwrite("focus_image.jpg", focus_image)

    if has_res:
        # YOLO检测
        res_img, yolo_list = self_yolo.detect(focus_image)
        cv2.imwrite("res_img.jpg", res_img)

        img_shape = res_img.shape

        # 坐标换算
        pixel_list = yolo_to_pixel(yolo_list, res_img.shape[0], res_img.shape[1])
        coordinate_list = coordinate_mapping(pixel_list, WIDTH_GOBANG, LENGTH_GOBANG, img_shape[0], img_shape[1])
        pos_set, ai_pos_set, pos_set_conf, ai_pos_set_conf = coordinate_to_pos(coordinate_list, go_stones)

        yolo_data = {
            "black_pieces": list(pos_set - ai_pos_set),
            "white_pieces": list(ai_pos_set) if go_stones == "black" else list(pos_set - ai_pos_set),
            "black_conf": pos_set_conf,
            "white_conf": ai_pos_set_conf
        }
        # 发送YOLO数据到Qt界面
        try:
            yolo_socket.sendto(json.dumps(yolo_data).encode('utf-8'), (YOLO_UDP_IP, YOLO_UDP_PORT))
            print(f"发送YOLO数据到Qt界面: {yolo_data}")
        except socket.error as e:
            print(f"发送YOLO数据失败: {e}")

        if (ai_down_last not in ai_pos_set) and ai_down_last != (None, None):
            return ai_down_last[0], ai_down_last[1], None

        our_down_pos = find_last_down_pos(pos_set)
        print("玩家落子：", our_down_pos)

        if not our_down_pos:
            return None, None, None

        our_down_pos_x, our_down_pos_y = our_down_pos

        # 使用αβ剪枝算法生成下一步走法
        machine_pos = AB_optimize.alpha_beta_process(mod)
        if not machine_pos:
            print("无法生成下一步走法，可能是棋局已结束！")
            return None, None, None
        ai_down_pos_x, ai_down_pos_y = machine_pos
        print(f"AI推荐下一步走法：({ai_down_pos_x}, {ai_down_pos_y})")
        return our_down_pos, ai_down_pos_x, ai_down_pos_y, mod
    return None, None, None


def start_qt_app():
    """启动Qt界面"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # 不使用 sys.exit，确保主线程可控退出
    app.exec_()


if __name__ == '__main__':
    # 启动Qt界面线程
    qt_thread = threading.Thread(target=start_qt_app, daemon=True)
    qt_thread.start()
    time.sleep(1)  # 等待Qt界面初始化

    mod = '固若金汤'
    device = 'cpu'
    go_stones = "black"  # 机械臂执棋颜色
    status_now = 'start'
    model_path = get_root() + "/yolov5/runs/train/exp5/weights/best.pt"
    self_yolo = YoloDetecter(weights=model_path, device=device)

    capture = cv2.VideoCapture(0)
    pre_img = get_video_frame(capture)

    while detect_flag:
        cur_img = get_video_frame(capture)
        diff = cv2.absdiff(cur_img, pre_img)
        max_diff = np.max(diff)
        pre_img = cur_img
        cv2.imwrite(f"./images/pre_img.jpg", pre_img)

        if max_diff > 120:
            print(f"相邻两帧像素差异最大值大于一百二:{max_diff}")
            time.sleep(1)
            continue

        our_down_pos, ai_down_coordinate_x, ai_down_coordinate_y, mod = detct(pre_img, self_yolo, mod, go_stones,status_now)

        if Check() == 'white':
            status_now = 'White wins'
            print("白棋胜利！")
            break

        if Check() == 'black':
            status_now = 'Black wins'
            print("黑棋胜利！")
            break

        status_now = 'playing'

        if ai_down_coordinate_x is None:
            time.sleep(1)
            continue

        ql_main(ai_down_coordinate_x, ai_down_coordinate_y)
        ai_down_last = (ai_down_coordinate_x, ai_down_coordinate_y)
        time.sleep(20)  # 模拟机械臂执行时间

    # 清理资源
    yolo_socket.close()
    capture.release()
    cv2.destroyAllWindows()