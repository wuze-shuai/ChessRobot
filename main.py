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
# import Alpha_beta_optimize
import AB_optimize
import Alpha_beta_optimize
from ql_main import ql_main
from yolov5.detect_self import YoloDetecter
from image_find_focus import FocusFinder
# from finde_focus import FocusFinder
from tools import coordinate_mapping, coordinate_to_pos, pos_to_coordinate, Check, get_video_frame, safe_detect, \
                get_current_boards, compare_boards, play_sound
import websockets
import asyncio
import json
from chess_qt import *
import socket
# 导入Qt界面（假设robot_arm.py在同一目录）
from qt_YOLO import MainWindow
from pathlib import Path
from ds_client import *
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
HIGH_GOBANG = Global_variables.HIGH_GOBANG  # 五子棋棋盘+棋子高度
WIDTH_ERR_GOBANG = Global_variables.WIDTH_ERR_GOBANG  # 五子棋盘内外边框间距(宽度方向)
LENGTH_ERR_GOBANG = Global_variables.LENGTH_ERR_GOBANG  # 五子棋盘内外边框间距(长度方向)
ROW_GOBANG = Global_variables.ROW_GOBANG  # 五子棋盘行数(宽度方向)
COLUMN_GOBANG = Global_variables.COLUMN_GOBANG  # 五子棋盘列数(长度方向)


# 玩家历史落子
history_set = set()

# ai上一步落子位置
ai_down_last = (None, None)


def find_last_down_pos(now_pos_set):
    global history_set
    our_down_pos = now_pos_set - history_set
    if len(our_down_pos) == 0:
        print("该你下了哦！")
        # time_lag = time.time() - Global_variables.start_time
        # if time_lag > Global_variables.remind_time:
        #     play_sound("your")
        #     Global_variables.start_time = time.time()
        return None
    
    if len(our_down_pos) > 1:
        print("你下了%d个棋子，不许耍赖哦!" % (len(our_down_pos)))
        return None

    # LOG.info(f"玩家最后一次落子落子位置:{our_down_pos}")
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
    for x, y, w, h,conf, c in yolo_list:
        pixel_y = y * cols_b
        pixel_x = x * rows_b
        data.append([pixel_x, pixel_y,conf, c])
    return data

def coordinate_to_pos(coordinate_list, go_stones):
    pos_set = set()
    ai_pos_set = set()
    ai_pos_set_conf = list()
    pos_set_conf = list()
    if go_stones == "white":
        player_class = 1
    else:
        player_class = 0

    for coordinate_x, coordinate_y, conf, c in coordinate_list:
        pos_x = round(
            abs(coordinate_x - WIDTH_ERR_GOBANG) / (WIDTH_GOBANG - 2 * WIDTH_ERR_GOBANG) * (ROW_GOBANG - 1))
        pos_y = round(
            abs(coordinate_y - LENGTH_ERR_GOBANG) / (LENGTH_GOBANG - 2 * LENGTH_ERR_GOBANG) * (COLUMN_GOBANG - 1))

        if c == player_class:  # 计算玩家的棋子
            pos_set.add((pos_x, pos_y))
            pos_set_conf.append((pos_x, pos_y, float(conf)))
        else:
            ai_pos_set.add((pos_x, pos_y))
            ai_pos_set_conf.append((pos_x, pos_y, float(conf)))
    return pos_set, ai_pos_set, pos_set_conf, ai_pos_set_conf

def coordinate_mapping(pixel_list, physical_rows, physical_cols, pixel_rows, pixel_cols):
    # LOG.debug(f"坐标映射: {physical_rows}, {physical_cols}, {pixel_rows}, {pixel_cols}")
    data = []
    for x, y, conf, c in pixel_list:
        x = x * physical_rows / pixel_rows
        y = y * physical_cols / pixel_cols
        data.append([x, y, conf, c])
    return data


def detct(image, self_yolo, mod, go_stones,status_now):
    global ai_down_last,text1
    # 图像校正和目标区域提取
    focus_finder = FocusFinder()
    focus_image, has_res = focus_finder.find_focus(image)
    cv2.imwrite(f"images/focus_image.jpg", focus_image)

    if has_res:
        # yolo检测
        res_img, yolo_list = self_yolo.detect(focus_image)
        rotated_img = cv2.rotate(res_img, cv2.ROTATE_90_CLOCKWISE)
        # 保存旋转后的图像
        cv2.imwrite(f"images/res_img.jpg", rotated_img)
        print('yolo_list:',yolo_list)
        # cv2.imwrite("res_img.jpg", res_img)

        img_shape = res_img.shape

        # 坐标换算
        pixel_list = yolo_to_pixel(yolo_list, res_img.shape[0], res_img.shape[1])       # 坐标信息转换为像素[pixel_x, pixel_y, c]
        print('pixel_list:',pixel_list)
        coordinate_list = coordinate_mapping(pixel_list, WIDTH_GOBANG, LENGTH_GOBANG, img_shape[0], img_shape[1])       # 转换为针对棋盘的物理坐标
        print('coordinate_list:',coordinate_list)
        pos_set, ai_pos_set, pos_set_conf, ai_pos_set_conf = coordinate_to_pos(coordinate_list, go_stones)        # 转换为棋盘行列坐标
        cv2.imwrite(f"images/res_img_{len(pixel_list)}.jpg", res_img)
        # 判断ai最后一次落子是否成功
        if (ai_down_last not in ai_pos_set) and ai_down_last != (None, None):
            return pos_set, ai_down_last[0], ai_down_last[1], mod

        # 记录 yolo_data 到日志文件
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)  # 创建 logs 目录（如果不存在）
        log_file = log_dir / "yolo_log.json"
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "ai_pos_set": list(ai_pos_set),
            "pos_set": list(pos_set),
        }
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n')  # 每条记录占一行，便于阅读和解析
        except Exception as e:
            print(f"写入日志文件失败: {e}")

        # 发送棋子位置到Qt界面
        yolo_data = {
            "black_pieces": list(ai_pos_set) if go_stones == "black" else list(pos_set),  # 玩家棋子
            "white_pieces": list(pos_set) if go_stones == "black" else list(ai_pos_set),  # AI棋子
            "black_conf": list(ai_pos_set_conf) if go_stones == "black" else list(pos_set_conf),
            "white_conf": list(pos_set_conf) if go_stones == "black" else list(ai_pos_set_conf),
        }
        try:
            yolo_socket.sendto(json.dumps(yolo_data).encode('utf-8'), (YOLO_UDP_IP, YOLO_UDP_PORT))
            # print(f"发送YOLO数据到Qt界面: {yolo_data}")
        except Exception as e:
            print(f"发送YOLO数据失败: {e}")

         # 计算玩家最后一次落子
        our_down_pos = find_last_down_pos(pos_set)
        print("玩家落子：", our_down_pos)

        if our_down_pos:
            our_down_pos_x, our_down_pos_y = our_down_pos
            record_pos((our_down_pos_x, our_down_pos_y), go_stones)
            result = Check()
            if result not in ('white','black'):
                if text1 == 0:
                    play_sound("该我下了", 'mp3')
                    text1 = 1
                elif text1 == 1:
                    play_sound("让我想想，再下哪一步呢", 'mp3')
                    text1 = 2
                elif text1 == 2:
                    text1 = 3
                else:
                    play_sound("看我下吧", 'mp3')
                    text1 = 0
            else:
                return None, None, None, mod
        else:
            return None, None, None, mod


        #大模型算法
        response_data = asyncio.run(send_yolo_result(pos_set, ai_pos_set, our_down_pos_x, our_down_pos_y, go_stones,status_now,mod))
        if "error" in response_data:
            print('mod:',mod)
            # αβ剪枝算法
            ai_down_pos_x, ai_down_pos_y = human_vs_machine(mod, (our_down_pos_x, our_down_pos_y), go_stones)
        else:
            print('mod:', mod)
            ai_down_pos_x, ai_down_pos_y, reason , mod = parase_response(response_data)
            ai_down_pos_x, ai_down_pos_y = human_vs_machine(mod, (our_down_pos_x, our_down_pos_y), go_stones)

            # 记录下棋位置：
            record_pos((our_down_pos_x, our_down_pos_y), go_stones, ai_down_pos_x, ai_down_pos_y)

        # if ai_down_pos_x == None or ai_down_pos_y == None:
        print("ai落子棋盘格坐标：", (ai_down_pos_x, ai_down_pos_y))

        return our_down_pos, ai_down_pos_x, ai_down_pos_y, mod
    
    return None, None, None, mod

def record_pos(pos, go_stones,ai_down_pos_x=None, ai_down_pos_y=None):
    x1, y1 = pos
    if go_stones == "white":
        Global_variables.black[x1][y1] = 1
    else:
        Global_variables.white[x1][y1] = 1
    Global_variables.flag[x1][y1] = 1
    #仅更新玩家棋子
    if ai_down_pos_x==None and ai_down_pos_y==None:
        pass
    else:
        i = ai_down_pos_x
        j = ai_down_pos_y

        if go_stones == "white":
            Global_variables.white[i][j] = 1
        else:
            Global_variables.black[i][j] = 1
        Global_variables.flag[i][j] = 1

def human_vs_machine(mod, pos, go_stones):
    x1, y1 = pos
    if go_stones == "white":
        Global_variables.black[x1][y1] = 1
    else:
        Global_variables.white[x1][y1] = 1
    Global_variables.flag[x1][y1] = 1
    machine_pos = Alpha_beta_optimize.alpha_beta_process(mod)

    # machine_pos, white_max_score, black_max_score = AB_optimize.alpha_beta_process(mod,go_stones)
    # print("white_max_score", white_max_score)
    # print("black_max_score", black_max_score)
    # if go_stones == "white":
    #     if (white_max_score > black_max_score+2000 and white_max_score >8000):
    #         play_sound("您这步下的真巧", 'mp3')
    # if go_stones == "black":
    #     if (white_max_score < black_max_score+2000 and white_max_score >8000):
    #         play_sound("您这步下的真巧", 'mp3')

    if not machine_pos:
        print('机器对战已结束...')

    i = machine_pos[0]
    j = machine_pos[1]

    if go_stones == "white":
        Global_variables.white[i][j] = 1
    else:
        Global_variables.black[i][j] = 1

    # 测试图显示bug修复
    Global_variables.flag[i][j] = 1
    return machine_pos


def play_win_sound(winner, go_stones):
    if winner == go_stones:
        return "robot_win"
    else:
        return "win"

# WebSocket 客户端设置
ws = None

# 处理服务器返回的指令
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

            # 解析 next_move，例如 "move_to_x:5,y:5"
            if next_move.startswith("move_to_x:"):
                try:
                    move_data = next_move.replace("move_to_x:", "").split(",y:")
                    target_x = float(move_data[0])
                    target_y = float(move_data[1])
                    print(f"解析指令: 移动 {color} 棋子到 ({target_x}, {target_y})")
                    # 更新 AI 下棋坐标（可调用 ql_main 或机器人控制）
                    Global_variables.ai_down_last = (target_x, target_y)
                    ql_main(target_x, target_y)  # 调用 AI 下棋逻辑
                    # robot_move_to(target_x, target_y, color)  # 如果有机器人控制
                except (IndexError, ValueError) as e:
                    print(f"解析指令失败: {e}")
    except json.JSONDecodeError:
        print("收到无效的 JSON 消息")


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
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    # 在单独线程中运行 WebSocket
    threading.Thread(target=ws.run_forever, daemon=True).start()

# 发送 YOLO 检测结果到服务器
async def send_yolo_result(pos_set, ai_pos_set, x, y, ai_color,status_now,mod):
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            # 构建 YOLO 数据
            data = {
                'ai_pos_set': list(ai_pos_set),
                'pos_set': list(pos_set),
                "x": x,
                "y": y,
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

    # mod = '中等'
    mod = '固若金汤'
    device = 'cpu'
    go_stones = "black"     # 机械臂执棋颜色
    text1 = 1
    #设置阶段：
    status_now = 'start'
    model_path = get_root() + "/yolov5/runs/train/exp5/weights/best.pt"
    self_yolo = YoloDetecter(weights=model_path, device=device)

    capture = cv2.VideoCapture(0)
    last_class_list = []
    focus_finder = FocusFinder()

    # 启动 WebSocket 客户端
    start_websocket()
    time.sleep(1)  # 等待 WebSocket 连接建立
    # play_sound("开始下棋啦", "mp3")
    if go_stones == "black":
        Global_variables.flag[6][6] = 1
        Global_variables.black[6][6] = 1
        ai_down_last = (6, 6)
        ql_main(6,6)
        time.sleep(20)

    Global_variables.start_time = time.time()
    pre_img = get_video_frame(capture)
    time1 = None
    play_time = 0
    while detect_flag:
        time2 = time.time()  # 记录循环开始时间
        if time1 is not None and (time2 - time1) > 8 and play_time == 0:  # 检查时间差
            play_sound("该你下啦", "mp3")
            play_time = 1
        if time1 is not None and (time2 - time1) > 15 and play_time ==1 :  # 检查时间差
            play_sound("玩家8秒不下提示---我等的花儿都谢了", "mp3")
            play_time = 2
        cur_img = get_video_frame(capture)
        diff = cv2.absdiff(cur_img, pre_img)
        max_diff = np.max(diff)
        pre_img = cur_img
        cv2.imwrite(f"./images/pre_img.jpg", pre_img)

        if max_diff > 120:
            print(f"相邻两帧像素差异最大值大于一百二:{max_diff}")
            time.sleep(1)
            continue

        our_down_pos , ai_down_coordinate_x, ai_down_coordinate_y ,mod= detct(pre_img, self_yolo, mod, go_stones, status_now)

        if Check() == 'white':
            status_now = 'White wins'
            if go_stones == "white":
                play_sound("你输了", "mp3")
            else:
                play_sound("你赢了", "mp3")
            break

        if Check() == 'black':
            status_now = 'Black wins'
            if go_stones == "black":
                ql_main(ai_down_coordinate_x, ai_down_coordinate_y)
                time.sleep(20)
                play_sound("你输了", "mp3")
            else:
                play_sound("你赢了", "mp3")
            break

        status_now = 'playing'
        
        if ai_down_coordinate_x == None:
            time.sleep(1)
            continue

        ql_main(ai_down_coordinate_x, ai_down_coordinate_y)
        ai_down_last = (ai_down_coordinate_x, ai_down_coordinate_y)
        time.sleep(20)
        time1 = time.time()  # 记录执行 ql_main 后的时间
        play_time =0
        Global_variables.start_time = time.time()
    # 清理资源
    yolo_socket.close()  