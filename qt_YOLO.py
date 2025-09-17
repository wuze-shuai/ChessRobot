import sys
import time
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import socket
import json
from pathlib import Path
import serial

import Global_variables

class GobangBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.grid_size = 40
        self.board_size = 13
        self.selected_pos = (0, 0)
        self.black_pieces = {}  # 存储 { (x, y): conf }
        self.white_pieces = {}  # 存储 { (x, y): conf }
        self.history = []  # 存储 [(x, y, color, conf, timestamp)]
        self.setFixedSize(self.board_size * self.grid_size, self.board_size * self.grid_size + 10)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制棋盘
        painter.setPen(QPen(Qt.black, 2))
        painter.drawEllipse(QPoint(int(self.board_size * self.grid_size / 2), 10), 8, 8)
        painter.drawText(int(self.board_size * self.grid_size / 2) + 15, 15, "机械臂位置")
        for i in range(self.board_size):
            painter.drawLine(self.grid_size // 2, self.grid_size // 2 + i * self.grid_size + 20,
                             self.width() - self.grid_size // 2, self.grid_size // 2 + i * self.grid_size + 20)
            painter.drawLine(self.grid_size // 2 + i * self.grid_size, self.grid_size // 2 + 20,
                             self.grid_size // 2 + i * self.grid_size, self.height() - self.grid_size // 2 + 10)

            if i == (self.board_size - 1):
                painter.drawText(self.grid_size // 2 + (self.board_size - 1) * self.grid_size + 10,
                                 self.grid_size // 2 + i * self.grid_size + 5 + 20, "X")
            else:
                painter.drawText(self.grid_size // 2 + (self.board_size - 1) * self.grid_size + 10,
                                 self.grid_size // 2 + i * self.grid_size + 5 + 20, str(i))

            if i == 0:
                painter.drawText(self.grid_size // 2 + i * self.grid_size - 5, 15 + 20, "Y")
            else:
                painter.drawText(self.grid_size // 2 + i * self.grid_size - 5, 15 + 20, str((self.board_size - 1) - i))

        # 绘制黑棋
        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(QBrush(Qt.black))
        for (x, y), _ in self.black_pieces.items():
            center_x = self.grid_size // 2 + ((self.board_size - 1) - y) * self.grid_size
            center_y = self.grid_size // 2 + x * self.grid_size + 20
            painter.drawEllipse(QPoint(center_x, center_y), 10, 10)

        # 绘制白棋
        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(QBrush(Qt.white))
        for (x, y), _ in self.white_pieces.items():
            center_x = self.grid_size // 2 + ((self.board_size - 1) - y) * self.grid_size
            center_y = self.grid_size // 2 + x * self.grid_size + 20
            painter.drawEllipse(QPoint(center_x, center_y), 10, 10)

    def update_chess_state(self, black_positions, white_positions, black_conf, white_conf, history_list):
        """更新黑白棋子位置和置信度，并记录历史"""
        try:
            # 转换为字典：{(x, y): conf}
            black_conf_dict = {(int(x), int(y)): float(conf) for x, y, conf in black_conf}
            white_conf_dict = {(int(x), int(y)): float(conf) for x, y, conf in white_conf}

            # 验证坐标和置信度
            for x, y in black_positions | white_positions:
                if not (0 <= x < self.board_size and 0 <= y < self.board_size):
                    raise ValueError(f"无效坐标 ({x}, {y})，必须在0到{self.board_size - 1}之间")
            for (x, y), conf in list(black_conf_dict.items()) + list(white_conf_dict.items()):
                if not (0 <= conf <= 1):
                    raise ValueError(f"无效置信度 {conf}，必须在0到1之间")

            # 检查重叠
            if black_positions & white_positions:
                raise ValueError("黑棋和白棋位置不能重叠")
            if (black_positions & set(self.black_pieces.keys()) - set(black_conf_dict.keys())) or \
               (black_positions & set(self.white_pieces.keys()) - set(white_conf_dict.keys())) or \
               (white_positions & set(self.black_pieces.keys()) - set(black_conf_dict.keys())) or \
               (white_positions & set(self.white_pieces.keys()) - set(white_conf_dict.keys())):
                raise ValueError("新棋子位置已被占用")

            # 验证置信度数据匹配
            if black_positions != set(black_conf_dict.keys()):
                raise ValueError("黑棋位置与置信度数据不匹配")
            if white_positions != set(white_conf_dict.keys()):
                raise ValueError("白棋位置与置信度数据不匹配")

            # 更新棋子
            self.black_pieces.update(black_conf_dict)
            self.white_pieces.update(white_conf_dict)

            # 更新历史记录
            self.history.extend(history_list)
            self.history.sort(key=lambda x: x[4])  # 按timestamp排序
            self.repaint()
            return True, "棋子状态更新成功"
        except ValueError as e:
            return False, f"更新棋子状态失败: {str(e)}"

    def clear_chess_state(self):
        """清除棋盘上的黑白棋子和历史记录"""
        self.black_pieces.clear()
        self.white_pieces.clear()
        self.history.clear()
        self.repaint()
        return True, "棋盘已清除"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("棋局状态实时展示界面")
        self.resize(400, 580)
        self.ESP32_IP = Global_variables.ESP32_IP
        self.ESP32_PORT = 8081
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_socket.connect((self.ESP32_IP, self.ESP32_PORT))

        self.yolo_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.yolo_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.yolo_socket.bind(('0.0.0.0', 5005))
            # print("UDP服务器绑定成功：0.0.0.0:5005")
        except Exception as e:
            print(f"UDP绑定失败: {e}")
            sys.exit(1)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 棋盘面板
        display_panel = QWidget()
        display_layout = QHBoxLayout(display_panel)
        self.gobang_board = GobangBoard()
        display_layout.addWidget(self.gobang_board)
        layout.addWidget(display_panel)

        # 历史记录面板
        self.history_list = QListWidget()
        self.history_list.setFixedHeight(150)
        layout.addWidget(self.history_list)

        self.check_coordinate(Global_variables.filename, 0, 0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.receive_yolo_data)
        self.timer.start(100)
        print("QTimer启动，检查间隔100ms")

    def receive_yolo_data(self):
        """接收YOLO发送的棋子位置和置信度数据"""
        try:
            self.yolo_socket.settimeout(0.01)
            data, addr = self.yolo_socket.recvfrom(1024)
            # print(f"收到YOLO数据来自 {addr}: {data.decode('utf-8')}")
            yolo_data = json.loads(data.decode('utf-8'))

            black_positions = {(int(x), int(y)) for [x, y] in yolo_data.get("black_pieces", [])}
            white_positions = {(int(x), int(y)) for [x, y] in yolo_data.get("white_pieces", [])}
            black_conf = [(int(x), int(y), float(conf)) for [x, y, conf] in yolo_data.get("black_conf", [])]
            white_conf = [(int(x), int(y), float(conf)) for [x, y, conf] in yolo_data.get("white_conf", [])]
            # print(f"解析黑棋: {black_positions}, 白棋: {white_positions}, 黑棋置信度: {black_conf}, 白棋置信度: {white_conf}")

            # 构建历史记录（新棋子）
            history_list = []
            timestamp = time.time()
            for x, y, conf in black_conf:
                if (x, y) not in set(self.gobang_board.black_pieces.keys()) | set(self.gobang_board.white_pieces.keys()):
                    history_list.append((x, y, "黑", conf, timestamp))
            for x, y, conf in white_conf:
                if (x, y) not in set(self.gobang_board.black_pieces.keys()) | set(self.gobang_board.white_pieces.keys()):
                    history_list.append((x, y, "白", conf, timestamp))

            # 更新棋盘和历史
            success, message = self.update_chess_state(black_positions, white_positions, black_conf, white_conf, history_list)
            print(f"棋盘更新结果: {message}")
            if success:
                # 更新历史列表显示
                for x, y, color, conf, _ in history_list:
                    self.history_list.addItem(f"({x}, {y}) {color} {conf*100:.0f}%")
            else:
                print(f"更新失败详情: {message}")
        except socket.timeout:
            pass
        except json.JSONDecodeError as e:
            print(f"YOLO数据格式错误: {e}")
        except Exception as e:
            print(f"YOLO数据处理错误: {e}")

    def update_chess_state(self, black_positions, white_positions, black_conf, white_conf, history_list):
        return self.gobang_board.update_chess_state(black_positions, white_positions, black_conf, white_conf, history_list)

    def clear_chess_board(self):
        self.history_list.clear()
        return self.gobang_board.clear_chess_state()

    def set_arm_angles(self, angle1, angle2, angle3):
        try:
            self.axis1_value = f"{float(angle1):.2f}"
            self.axis2_value = f"{float(angle2):.2f}"
            self.axis3_value = f"{float(angle3):.2f}"
            return True, "角度值设置成功"
        except ValueError:
            return False, "请输入有效的数字"

    def check_coordinate(self, file_path, x, y):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            angles = data.get("angles", {})
            key = f"{x},{y}"
            if key in angles:
                a, b, c = map(float, angles[key].split('/'))
                self.axis1_value = f"{a:.2f}"
                self.axis2_value = f"{b:.2f}"
                self.axis3_value = f"{c:.2f}"
            else:
                # print(f"坐标({x},{y})不存在")
                pass

            chess_state = data.get("chess_state", {})
            black_pieces = {(x, y) for [x, y] in chess_state.get("black_pieces", [])}
            white_pieces = {(x, y) for [x, y] in chess_state.get("white_pieces", [])}
            black_conf = [(x, y, 1.0) for [x, y] in chess_state.get("black_pieces", [])]
            white_conf = [(x, y, 1.0) for [x, y] in chess_state.get("white_pieces", [])]
            history_list = [(x, y, "黑", 1.0, time.time()) for x, y in black_pieces] + \
                          [(x, y, "白", 1.0, time.time()) for x, y in white_pieces]
            self.update_chess_state(black_pieces, white_pieces, black_conf, white_conf, history_list)
        except FileNotFoundError:
            print("文件未找到")
        except json.JSONDecodeError:
            print("文件格式错误")

    def update_txt_file(self, data_dict, file_path):
        file = Path(file_path)
        existing_data = {}
        if file.exists():
            try:
                with open(file, 'r') as f:
                    existing_data = json.loads(f.read())
            except json.JSONDecodeError:
                existing_data = {}

        existing_data.update(data_dict)
        with open(file, 'w') as f:
            f.write(json.dumps(existing_data, indent=4))

    def closeEvent(self, event):
        self.yolo_socket.close()
        self.client_socket.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())