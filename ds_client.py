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
