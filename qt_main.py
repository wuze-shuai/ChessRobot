# -*- coding: utf-8 -*-
import time
import cv2
import websocket
import threading
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import Global_variables
import Alpha_beta_optimize
from ql_main import ql_main
from yolov5.detect_self import YoloDetecter
from image_find_focus import FocusFinder
from tools import coordinate_mapping, coordinate_to_pos, pos_to_coordinate, Check, get_video_frame, safe_detect, \
    get_current_boards, compare_boards
import websockets
import asyncio
import json
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import socket

# 导入Qt界面（假设robot_arm.py在同一目录）
from qt_YOLO import MainWindow

detect_flag = True

WIDTH_GOBANG = Global_variables.WIDTH_GOBANG  # 五子棋盘总宽度
LENGTH_GOBANG = Global_variables.LENGTH_GOBANG  # 五子棋盘总长度
HIGH_GOBANG = Global_variables.HIGH_GOBANG  # 五子棋棋盘+棋子高度
WIDTH_ERR_GOBANG = Global_variables.WIDTH_ERR_GOBANG  # 五子棋盘内外边框间距(宽度方向)
LENGTH_ERR_GOBANG = Global_variables.LENGTH_ERR_GOBANG  # 五子棋盘内外边框间距(长度方向)
ROW_GOBANG = Global_variables.ROW_GOBANG  # 五子棋盘行数(宽度方向)
COLUMN_GOBANG = Global_variables.COLUMN_GOBANG  # 五子棋盘列数(长度方向)

# 玩家历史落子
history_set = set()

# AI上一步落子位置
ai_down_last = (None, None)

# UDP客户端用于发送YOLO数据到Qt界面
YOLO_UDP_IP = "127.0.0.1"
YOLO_UDP_PORT = 5005
yolo_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


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
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
        Img = Image.fromarray(cv2image)  # 将图像转换成Image对象
    return Img


def get_root():
    file = Path(__file__).resolve()
    parent_dir = file.parent
    root = str(parent_dir).replace("\\", "/")
    return root


def yolo_to_pixel(yolo_list, rows_b, cols_b):
    data = []
    for x, y, w, h, c in yolo_list:
        pixel_y = y * cols_b
        pixel_x = x * rows_b
        data.append([pixel_x, pixel_y, c])
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
        pos_set, ai_pos_set = coordinate_to_pos(coordinate_list, go_stones)

        # 发送棋子位置到Qt界面
        yolo_data = {
            "black_pieces": list(pos_set - ai_pos_set),  # 玩家棋子
            "white_pieces": list(ai_pos_set) if go_stones == "black" else list(pos_set - ai_pos_set),  # AI棋子
        }
        try:
            yolo_socket.sendto(json.dumps(yolo_data).encode('utf-8'), (YOLO_UDP_IP, YOLO_UDP_PORT))
            print(f"发送YOLO数据到Qt界面: {yolo_data}")
        except Exception as e:
            print(f"发送YOLO数据失败: {e}")

        # 判断AI最后一次落子是否成功
        if (ai_down_last not in ai_pos_set) and ai_down_last != (None, None):
            return ai_down_last[0], ai_down_last[1], None

        # 计算玩家最后一次落子
        our_down_pos = find_last_down_pos(pos_set)
        print("玩家落子：", our_down_pos)

        if not our_down_pos:
            return None, None, None

        our_down_pos_x, our_down_pos_y = our_down_pos
        # 大模型算法
        response_data = asyncio.run(
            send_yolo_result(pos_set, ai_pos_set, our_down_pos_x, our_down_pos_y, go_stones, status_now))
        ai_down_pos_x, ai_down_pos_y, reason = parase_response(response_data)
        if ai_down_pos_x is None or ai_down_pos_y is None:
            # α-β剪枝算法
            ai_down_pos_x, ai_down_pos_y = human_vs_machine(mod, (our_down_pos_x, our_down_pos_y), go_stones)
        print("AI落子棋盘格坐标：", (ai_down_pos_x, ai_down_pos_y))

        return our_down_pos, ai_down_pos_x, ai_down_pos_y

    return None, None, None


def human_vs_machine(mod, pos, go_stones):
    x1, y1 = pos
    if go_stones == "white":
        Global_variables.black[x1][y1] = 1
    else:
        Global_variables.white[x1][y1] = 1
    Global_variables.flag[x1][y1] = 1

    machine_pos = Alpha_beta_optimize.alpha_beta_process(mod)
    if not machine_pos:
        print('机器对战已结束...')

    i = machine_pos[0]
    j = machine_pos[1]

    if go_stones == "white":
        Global_variables.white[i][j] = 1
    else:
        Global_variables.black[i][j] = 1

    Global_variables.flag[i][j] = 1
    return machine_pos


def play_win_sound(winner, go_stones):
    if winner == go_stones:
        return "robot_win"
    else:
        return "win"


# WebSocket 客户端设置
ws = None


def on_message(ws, message):
    try:
        data = json.loads(message)
        if "error" in data:
            print(f"错误: {data['error']}")
        else:
            next_move = data.get("next_move", "未知指令")
            color = data.get("color", "unknown")
            timestamp = data.get("timestamp", time.time())
            print(f"收到服务器指令: {next_move} (颜色: {color}, 时间: {timestamp})")

            if next_move.startswith("move_to_x:"):
                try:
                    move_data = next_move.replace("move_to_x:", "").split(",y:")
                    target_x = float(move_data[0])
                    target_y = float(move_data[1])
                    print(f"解析指令: 移动 {color} 棋子到 ({target_x}, {target_y})")
                    Global_variables.ai_down_last = (target_x, target_y)
                    ql_main(target_x, target_y)
                except (IndexError, ValueError) as e:
                    print(f"解析指令失败: {e}")
    except json.JSONDecodeError:
        print("收到无效的 JSON 消息")


def on_error(ws, error):
    print(f"WebSocket 错误: {error}")


def on_close(ws, close_status_code, close_msg):
    print("WebSocket 连接关闭，正在尝试重连...")
    time.sleep(5)
    start_websocket()


def on_open(ws):
    print("WebSocket 连接已建立")


def start_websocket():
    global ws
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        "ws://localhost:8765",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    threading.Thread(target=ws.run_forever, daemon=True).start()


async def send_yolo_result(pos_set, ai_pos_set, x, y, color, status_now):
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            data = {
                'ai_pos_set': list(ai_pos_set),
                'pos_set': list(pos_set),
                "x": x,
                "y": y,
                "color": color,
                'status_now': status_now,
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(data))
            print(f"发送 YOLO 数据: {data}")
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"收到服务器响应: {response_data}")
            return response_data
    except Exception as e:
        print(f"WebSocket 错误: {e}")
        return {"error": str(e)}


def parase_response(response_data):
    x1 = response_data.get("x1", None)
    y1 = response_data.get("y1", None)
    reason = response_data.get("reason", "未提供原因")
    timestamp = response_data.get("timestamp", None)
    print(f"({x1}, {y1})")
    return x1, y1, reason


def start_qt_app():
    """启动Qt界面"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # 启动Qt界面线程
    qt_thread = threading.Thread(target=start_qt_app, daemon=True)
    qt_thread.start()
    time.sleep(1)  # 等待Qt界面初始化

    # 初始化
    mod = '和我一样6的Level'
    device = 'cpu'
    go_stones = "black"  # 机械臂执棋颜色
    status_now = 'start'
    model_path = get_root() + "/yolov5/runs/train/exp5/weights/best.pt"
    self_yolo = YoloDetecter(weights=model_path, device=device)

    capture = cv2.VideoCapture(0)
    last_class_list = []
    focus_finder = FocusFinder()

    start_websocket()
    time.sleep(1)

    if go_stones == "black":
        Global_variables.flag[4][4] = 1
        Global_variables.black[4][4] = 1
        ai_down_last = (4, 4)
        response_data = asyncio.run(send_yolo_result(set(), {(4, 4)}, 4, 4, go_stones, status_now))
        x1, y1, reason = parase_response(response_data)
        ql_main(4, 4)
        time.sleep(20)

    Global_variables.start_time = time.time()
    pre_img = cv2.imread("pre_img.jpg")
    while detect_flag:
        cur_img = cv2.imread("pre_img.jpg")
        diff = cv2.absdiff(cur_img, pre_img)
        max_diff = np.max(diff)
        pre_img = cur_img
        cv2.imwrite("pre_img.jpg", pre_img)

        if max_diff > 120:
            print(f"相邻两帧像素差异最大值大于一百二:{max_diff}")
            time.sleep(1)
            continue

        our_down_pos, ai_down_coordinate_x, ai_down_coordinate_y = detct(pre_img, self_yolo, mod, go_stones, status_now)

        if ai_down_coordinate_x is None:
            time.sleep(1)
            continue

        print(ai_down_coordinate_x, ai_down_coordinate_y)

        if Check() == 'black':
            if go_stones == "black":
                ql_main(ai_down_coordinate_x, ai_down_coordinate_y)
                time.sleep(12)
            print('Black wins')
            break
        if Check() == 'white':
            print('White wins')
            break

        ql_main(ai_down_coordinate_x, ai_down_coordinate_y)
        ai_down_last = (ai_down_coordinate_x, ai_down_coordinate_y)
        time.sleep(20)

        Global_variables.start_time = time.time()

    # 清理资源
    yolo_socket.close()