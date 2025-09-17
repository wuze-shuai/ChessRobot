from time import sleep
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import Global_variables
import time
import AB_optimize
from yolov5.detect_self import YoloDetecter
from image_find_focus import FocusFinder
from tools import coordinate_mapping, coordinate_to_pos, get_video_frame
import json
import socket
from qt_YOLO import MainWindow
from PyQt5.QtWidgets import QApplication
import sys
import threading
import keyboard
import websocket
import websockets
import asyncio
import json
#机械臂控制函数
# from ql_main import ql_main

# UDP客户端用于发送YOLO数据到Qt界面
YOLO_UDP_IP = "127.0.0.1"
YOLO_UDP_PORT = 5005
yolo_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 棋盘参数
WIDTH_GOBANG = Global_variables.WIDTH_GOBANG
LENGTH_GOBANG = Global_variables.LENGTH_GOBANG
WIDTH_ERR_GOBANG = Global_variables.WIDTH_ERR_GOBANG
LENGTH_ERR_GOBANG = Global_variables.LENGTH_ERR_GOBANG
ROW_GOBANG = Global_variables.ROW_GOBANG
COLUMN_GOBANG = Global_variables.COLUMN_GOBANG


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
    pos_set = set()  # 玩家棋子
    ai_pos_set = set()  # AI
    # 棋子
    pos_set_conf = []
    ai_pos_set_conf = []
    player_class = 1 if go_stones == "white" else 0

    for coordinate_x, coordinate_y, conf, c in coordinate_list:
        pos_x = round(
            abs(coordinate_x - WIDTH_ERR_GOBANG) / (WIDTH_GOBANG - 2 * WIDTH_ERR_GOBANG) * (ROW_GOBANG - 1))
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


def initialize_board(pos_set, ai_pos_set, go_stones):
    """初始化棋盘状态"""
    Global_variables.black = [[0] * COLUMN_GOBANG for _ in range(ROW_GOBANG)]
    Global_variables.white = [[0] * COLUMN_GOBANG for _ in range(ROW_GOBANG)]
    Global_variables.flag = [[0] * COLUMN_GOBANG for _ in range(ROW_GOBANG)]

    # 记录玩家棋子
    for pos_x, pos_y in pos_set:
        if go_stones == "white":
            Global_variables.black[pos_x][pos_y] = 1
        else:
            Global_variables.white[pos_x][pos_y] = 1
        Global_variables.flag[pos_x][pos_y] = 1

    # 记录AI棋子
    for pos_x, pos_y in ai_pos_set:
        if go_stones == "white":
            Global_variables.white[pos_x][pos_y] = 1
        else:
            Global_variables.black[pos_x][pos_y] = 1
        Global_variables.flag[pos_x][pos_y] = 1

def on_error(ws, error):
    # print(f"WebSocket 错误: {error}")
    pass

def on_close(ws, close_status_code, close_msg):
    print("WebSocket 连接关闭，正在尝试重连...")
    time.sleep(5)
    start_websocket()


def on_open(ws):
    print("WebSocket 连接已建立")

# 启动 WebSocket 客户端
def start_websocket():
    global ws
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        "ws://localhost:8765",
        on_open=on_open,
        on_error=on_error,
        on_close=on_close
    )
    # 在单独线程中运行 WebSocket
    threading.Thread(target=ws.run_forever, daemon=True).start()

# 发送 YOLO 检测结果到服务器
async def send_yolo_result(pos_set, ai_pos_set,ai_color,status_now,mod):
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            # 构建 YOLO 数据
            data = {
                'ai_pos_set': list(ai_pos_set),
                'pos_set': list(pos_set),
                "color": ai_color,
                'status_now': status_now,
                'mod': mod,
                "timestamp": time.time()
            }
            # 发送数据
            await websocket.send(json.dumps(data))
            print(f"发送 YOLO 数据: {data}")

            # 接收服务器响应
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"收到服务器响应: {response_data}")
            return response_data
    except Exception as e:
        print(f"WebSocket 错误: {e}")
        return {"error": str(e)}

def parase_response(response_data):
    # 解析响应
    x1 = response_data.get("x1", None)
    y1 = response_data.get("y1", None)
    reason = response_data.get("reason", "未提供原因")
    mod = response_data.get("mod", None)
    # color = response_data.get("color", color)  # 回传的颜色
    timestamp = response_data.get("timestamp", None)

    return x1, y1, reason,mod

def detect_endgame(image, self_yolo, go_stones, mod):
    """检测残局并返回AI下一步走法"""
    # 图像校正和目标区域提取
    focus_finder = FocusFinder()
    focus_image, has_res = image, 1  # 临时使用输入图像，跳过棋盘检测

    if not has_res:
        print("未检测到棋盘！")
        return None, None

    # YOLO检测
    res_img, yolo_list = self_yolo.detect(focus_image)
    rotated_img = cv2.rotate(res_img, cv2.ROTATE_90_CLOCKWISE)

    # 保存旋转后的图像
    cv2.imwrite("res_img.jpg", rotated_img)
    # cv2.imwrite("res_img.jpg", res_img)

    img_shape = res_img.shape

    # 坐标转换
    pixel_list = yolo_to_pixel(yolo_list, res_img.shape[0], res_img.shape[1])
    coordinate_list = coordinate_mapping(pixel_list, WIDTH_GOBANG, LENGTH_GOBANG, img_shape[0], img_shape[1])
    pos_set, ai_pos_set, pos_set_conf, ai_pos_set_conf = coordinate_to_pos(coordinate_list, go_stones)
    # pos_set = [(7, 1), (5, 4), (6, 4), (1, 4), (7, 2), (6, 3)]
    # ai_pos_set = [(4, 4), (6, 2), (2, 4), (3, 4), (2, 3), (5, 3)]
    ai_pos_set= [(4, 4), (5, 5), (8, 4), (7, 7), (6, 5), (5, 4), (5, 7), (6, 7), (7, 6), (5, 6), (2, 2), (6, 6),(7, 5), (6, 3), (1, 3), (8, 5), (5, 2)]
    pos_set=[(7, 4), (2, 4), (8, 8), (3, 4), (2, 7), (4, 3),(5, 8), (4, 6), (6, 4), (2, 3), (4, 5), (3, 3), (2, 6), (4, 8), (5, 3), (2, 5), (4, 7)]
    # 统计黑白棋子个数并判断
    black_count = len(pos_set if go_stones == "black" else ai_pos_set)
    white_count = len(pos_set if go_stones == "white" else ai_pos_set)
    count_diff = abs(black_count - white_count)

    # 初始化棋盘状态
    initialize_board(pos_set, ai_pos_set, go_stones)

    # 发送棋子位置到Qt界面
    yolo_data = {
        "black_pieces": list(ai_pos_set) if go_stones == "black" else list(pos_set),
        "white_pieces": list(pos_set) if go_stones == "black" else list(ai_pos_set),
        "ai_next_move": None,
        "black_conf": list(ai_pos_set_conf) if go_stones == "black" else list(pos_set_conf),
        "white_conf": list(pos_set_conf) if go_stones == "black" else list(ai_pos_set_conf),
    }

    if count_diff <= 1:
        if black_count > white_count:
            ai_color = '黑棋'
        elif white_count > black_count:
            ai_color = '白棋'
        else:
            ai_color = '黑棋'
    else:
        print("棋盘棋子个数存在问题")

    # 调用Alpha-Beta剪枝算法
    # machine_pos = AB_optimize.alpha_beta_process(mod)
    # if not machine_pos:
    #     print("无法生成下一步走法，可能是棋局已结束！")
    #     yolo_data["ai_next_move"] = None
    # else:
    #     ai_down_pos_x, ai_down_pos_y = machine_pos
    #     print(f"AI推荐{ai_color}下一步走法：({ai_down_pos_x}, {ai_down_pos_y})")
    #     yolo_data["ai_next_move"] = [ai_down_pos_x, ai_down_pos_y]

    # 大模型算法
    status_now = 'playing'
    response_data = asyncio.run(
        send_yolo_result(pos_set, ai_pos_set,go_stones, status_now, mod))
    ai_down_pos_x, ai_down_pos_y, reason, mod = parase_response(response_data)
    yolo_data["ai_next_move"] = [ai_down_pos_x, ai_down_pos_y]

    # 发送包含AI走法的数据
    # try:
    #     yolo_socket.sendto(json.dumps(yolo_data).encode('utf-8'), (YOLO_UDP_IP, YOLO_UDP_PORT))
    #     print(f"发送初始YOLO数据到Qt界面: {yolo_data}")
    # except socket.error as e:
    #     print(f"发送AI走法数据失败: {e}")

    return ai_down_pos_x if ai_down_pos_x else None, ai_down_pos_y if ai_down_pos_y else None

def start_qt_app():
    """启动Qt界面"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()  # 不使用 sys.exit，确保线程可控退出

# WebSocket 客户端设置
ws = None

if __name__ == '__main__':
    # 启动Qt界面线程
    qt_thread = threading.Thread(target=start_qt_app, daemon=True)
    qt_thread.start()

    # 初始化
    mod = '中等'  # 难度模式
    device = 'cpu'
    go_stones = "black"  # AI执棋颜色
    model_path = get_root() + "/yolov5/runs/train/exp5/weights/best.pt"
    self_yolo = YoloDetecter(weights=model_path, device=device)
    # 启动 WebSocket 客户端
    start_websocket()
    time.sleep(1)  # 等待 WebSocket 连接建立
    # 获取摄像头图片
    # capture = cv2.VideoCapture(2)
    # image = get_video_frame(capture)
    # capture.release()
    # 读取本地棋谱
    img_analysis = r'E:\13project\03下棋机器人有线通讯\HM-BW\images\20250730_110836.jpg'
    image = cv2.imread(img_analysis)


    try:
        # 检测残局并获取AI走法
        ai_down_pos_x, ai_down_pos_y = detect_endgame(image, self_yolo, go_stones, mod)
        # 如果需要机械臂下棋，则取消下面的注释，并在开头导入ql_main函数
        # ql_main(ai_down_coordinate_x, ai_down_coordinate_y)

        # 等待用户按 'e' 键退出
        print("按 'e' 键退出程序...")
        keyboard.wait('e')
        print("检测到 'e' 键，程序退出")
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {e}")
    finally:
        # 发送终止信号
        try:
            shutdown_data = {"status": "shutdown"}
            yolo_socket.sendto(json.dumps(shutdown_data).encode('utf-8'), (YOLO_UDP_IP, YOLO_UDP_PORT))
            print(f"发送终止信号到Qt界面: {shutdown_data}")
        except socket.error as e:
            print(f"发送终止信号失败: {e}")
        # 清理资源
        yolo_socket.close()
        print("YOLO socket 已关闭")