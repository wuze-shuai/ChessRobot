import sys
import os
import time
import socket
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTabWidget,
                             QListWidget, QPushButton, QHBoxLayout, QLineEdit, QComboBox, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

import camera_data
from qt_YOLO import GobangBoard as YoloGobangBoard  # 导入 YOLO 的 GobangBoard，添加别名以区分
from game_analysis import (YoloDetecter, FocusFinder, yolo_to_pixel, coordinate_mapping, coordinate_to_pos,
                           initialize_board, send_yolo_result, parase_response, WIDTH_GOBANG, LENGTH_GOBANG,
                           YOLO_UDP_IP, YOLO_UDP_PORT)
import cv2
import asyncio

# 导入 robot_arm.py 中的 GobangBoard 和 MainWindow
from robot_arm import MainWindow as ArmMainWindow

# 导入 annotation_qt.py 中的 AutoAnnotationGUI
from annotation_qt import AutoAnnotationGUI

# 导入 admin_camera.py 中的 CameraInfoWindow
from admin_camera import CameraInfoWindow

from image_process import TunerWindow


from PyQt5.QtCore import QThread, pyqtSignal

class ABThread(QThread):
    """在子线程中运行AB算法，避免阻塞Qt主线程"""
    result_ready = pyqtSignal(tuple)   # 返回(x, y)坐标
    error_happened = pyqtSignal(str)   # 如果算法异常，发送错误信息

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def run(self):
        try:
            from game_analysis import AB_optimize
            machine_pos = AB_optimize.alpha_beta_process(self.mod)
            self.result_ready.emit(machine_pos)
        except Exception as e:
            self.error_happened.emit(str(e))

class DataCollectionPage(camera_data.CameraGUI):
    def __init__(self):
        super().__init__()
        # Customize the title
        self.setWindowTitle("数据收集模块")

class YOLOBoardWidget(QWidget):
    """YOLO 棋局监控小部件，复用 qt_YOLO.py 的逻辑，并集成 game_analysis.py 的分析功能"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # 棋盘面板 (复用 YoloGobangBoard)
        display_panel = QWidget()
        display_layout = QHBoxLayout(display_panel)
        self.gobang_board = YoloGobangBoard()
        display_layout.addWidget(self.gobang_board)
        layout.addWidget(display_panel)

        # 历史记录面板 (复用 QListWidget)
        self.history_list = QListWidget()
        self.history_list.setFixedHeight(150)
        layout.addWidget(self.history_list)

        # 配置面板（从 game_analysis.py 提取变量配置到 UI）
        config_panel = QWidget()
        config_layout = QHBoxLayout(config_panel)

        # 检测模型路径输入
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("YOLO 模型路径 (默认: yolov5/runs/train/exp5/weights/best.pt)")
        config_layout.addWidget(QLabel("模型路径:"))
        config_layout.addWidget(self.model_path_edit)
        browse_model_btn = QPushButton("打开文件")
        browse_model_btn.clicked.connect(self.browse_model)
        config_layout.addWidget(browse_model_btn)

        #对弈模型选择
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['AB', '大模型'])
        self.algorithm_combo.setCurrentText('AB')
        config_layout.addWidget(QLabel("算法:"))
        config_layout.addWidget(self.algorithm_combo)

        # 难度选择
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(['简单', '中等', '困难'])
        self.difficulty_combo.setCurrentText('中等')
        config_layout.addWidget(QLabel("难度:"))
        config_layout.addWidget(self.difficulty_combo)

        # AI 持棋颜色选择
        self.ai_color_combo = QComboBox()
        self.ai_color_combo.addItems(['black', 'white'])
        self.ai_color_combo.setCurrentText('black')
        config_layout.addWidget(QLabel("AI 颜色:"))
        config_layout.addWidget(self.ai_color_combo)

        # 图像源输入（路径或摄像头 0-5）
        self.image_source_edit = QLineEdit()
        self.image_source_edit.setPlaceholderText("图像源 (路径或摄像头 0-5)")
        config_layout.addWidget(QLabel("图像源:"))
        config_layout.addWidget(self.image_source_edit)
        browse_image_btn = QPushButton("打开文件")
        browse_image_btn.clicked.connect(self.browse_image_source)
        config_layout.addWidget(browse_image_btn)

        layout.addWidget(config_panel)

        # 分析按钮
        analyze_btn = QPushButton("分析棋局并获取 AI 走法")
        analyze_btn.clicked.connect(self.analyze_chess_board)
        layout.addWidget(analyze_btn)

        # 清除按钮
        clear_btn = QPushButton("清除棋盘")
        clear_btn.clicked.connect(self.clear_chess_board)
        layout.addWidget(clear_btn)

        # UDP socket 初始化 (复用 qt_YOLO.py)
        self.yolo_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.yolo_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.yolo_socket.bind(('0.0.0.0', 5005))
        except Exception as e:
            print(f"UDP绑定失败: {e}")
            sys.exit(1)

        # 定时器 (复用 qt_YOLO.py 的 timer)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.receive_yolo_data)
        self.timer.start(100)

    def browse_model(self):
        """打开文件对话框选择模型路径"""
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov5/runs/train/exp5/weights'))
        fname, _ = QFileDialog.getOpenFileName(self, "Select Model File", default_dir, "PT Files (*.pt)")
        if fname:
            self.model_path_edit.setText(fname)

    def browse_image_source(self):
        """打开文件对话框选择图像文件路径"""
        fname, _ = QFileDialog.getOpenFileName(self, "Select Image Source", "", "Images (*.jpg *.png *.bmp *.jpeg);;All Files (*)")
        if fname:
            self.image_source_edit.setText(fname)

    def receive_yolo_data(self):
        """接收 YOLO 数据 (直接复用/微调 qt_YOLO.py 的方法)"""
        try:
            self.yolo_socket.settimeout(0.01)
            data, addr = self.yolo_socket.recvfrom(1024)
            yolo_data = json.loads(data.decode('utf-8'))

            black_positions = {(int(x), int(y)) for [x, y] in yolo_data.get("black_pieces", [])}
            white_positions = {(int(x), int(y)) for [x, y] in yolo_data.get("white_pieces", [])}
            black_conf = [(int(x), int(y), float(conf)) for [x, y, conf] in yolo_data.get("black_conf", [])]
            white_conf = [(int(x), int(y), float(conf)) for [x, y, conf] in yolo_data.get("white_conf", [])]

            # 构建历史记录 (复用逻辑)
            history_list = []
            timestamp = time.time()
            for x, y, conf in black_conf:
                if (x, y) not in set(self.gobang_board.black_pieces.keys()) | set(
                        self.gobang_board.white_pieces.keys()):
                    history_list.append((x, y, "黑", conf, timestamp))
            for x, y, conf in white_conf:
                if (x, y) not in set(self.gobang_board.black_pieces.keys()) | set(
                        self.gobang_board.white_pieces.keys()):
                    history_list.append((x, y, "白", conf, timestamp))

            # 更新棋盘 (调用 GobangBoard 方法)
            success, message = self.gobang_board.update_chess_state(black_positions, white_positions, black_conf,
                                                                    white_conf, history_list)
            if success:
                for x, y, color, conf, _ in history_list:
                    self.history_list.addItem(f"({x}, {y}) {color} {conf * 100:.0f}%")
        except socket.timeout:
            pass
        except json.JSONDecodeError as e:
            print(f"YOLO数据格式错误: {e}")
        except Exception as e:
            print(f"YOLO数据处理错误: {e}")

    def clear_chess_board(self):
        """清除棋盘 (调用 GobangBoard 方法)"""
        self.history_list.clear()
        self.gobang_board.clear_chess_state()

    def analyze_chess_board(self):
        """集成 game_analysis.py 的 detect_endgame 功能，处理图像源并分析棋局"""
        # 获取用户配置
        model_path = self.model_path_edit.text() or os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "yolov5/runs/train/exp5/weights/best.pt")
        mod = self.difficulty_combo.currentText()
        go_stones = self.ai_color_combo.currentText()  # AI 持棋颜色
        image_source = self.image_source_edit.text()

        if not image_source:
            QMessageBox.warning(self, "错误", "请提供图像源（路径或摄像头 0-5）")
            return

        # 处理图像源：路径或摄像头
        try:
            if image_source.isdigit() and 0 <= int(image_source) <= 5:
                # 摄像头
                cap = cv2.VideoCapture(int(image_source))
                ret, image = cap.read()
                cap.release()
                if not ret:
                    raise ValueError("无法从摄像头读取图像")
            else:
                # 文件路径
                image = cv2.imread(image_source)
                if image is None:
                    raise ValueError("无法读取图像文件")
        except ValueError as e:
            QMessageBox.warning(self, "错误", str(e))
            return

        # 初始化 YOLO 模型（从 game_analysis.py 复用）
        self_yolo = YoloDetecter(weights=model_path, device='cpu')

        # 复用 game_analysis.py 的 detect_endgame 核心逻辑（略微调整以适应 UI）
        focus_finder = FocusFinder()
        focus_image = image  # 简化，假设已聚焦（可扩展）

        res_img, yolo_list = self_yolo.detect(focus_image)
        img_shape = res_img.shape

        pixel_list = yolo_to_pixel(yolo_list, res_img.shape[0], res_img.shape[1])
        coordinate_list = coordinate_mapping(pixel_list, WIDTH_GOBANG, LENGTH_GOBANG, img_shape[0], img_shape[1])
        pos_set, ai_pos_set, pos_set_conf, ai_pos_set_conf = coordinate_to_pos(coordinate_list, go_stones)

        # 统计棋子并判断颜色（复用逻辑）
        black_count = len(pos_set if go_stones == "black" else ai_pos_set)
        white_count = len(pos_set if go_stones == "white" else ai_pos_set)
        count_diff = abs(black_count - white_count)

        if count_diff > 1:
            QMessageBox.warning(self, "错误", "棋盘棋子个数存在问题")
            return

        ai_color = '黑棋' if (
                    black_count > white_count or (black_count == white_count and go_stones == 'black')) else '白棋'

        # 初始化棋盘（复用）
        initialize_board(pos_set, ai_pos_set, go_stones)

        algorithm = self.algorithm_combo.currentText()
        print('algorithm:', algorithm)

        def continue_with_move(ai_down_pos_x, ai_down_pos_y, reason=''):
            # 发送 YOLO 数据到 UDP（复用 game_analysis.py 逻辑）
            yolo_data = {
                "black_pieces": list(ai_pos_set) if go_stones == "black" else list(pos_set),
                "white_pieces": list(pos_set) if go_stones == "black" else list(ai_pos_set),
                "ai_next_move": [ai_down_pos_x, ai_down_pos_y] if ai_down_pos_x is not None else None,
                "black_conf": list(ai_pos_set_conf) if go_stones == "black" else list(pos_set_conf),
                "white_conf": list(pos_set_conf) if go_stones == "black" else list(ai_pos_set_conf),
            }
            try:
                yolo_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                yolo_socket.sendto(json.dumps(yolo_data).encode('utf-8'),
                                   (YOLO_UDP_IP, YOLO_UDP_PORT))
                print(f"发送 YOLO 数据: {yolo_data}")
                yolo_socket.close()
            except socket.error as e:
                print(f"发送失败: {e}")

            # 更新历史列表显示 AI 走法
            if ai_down_pos_x is not None:
                self.history_list.addItem(
                    f"AI 推荐 ({ai_down_pos_x}, {ai_down_pos_y}) {ai_color} - 原因: {reason}")
            else:
                self.history_list.addItem("无法生成下一步走法")

        if algorithm == 'AB':
            # 子线程运行AB算法
            self.ab_thread = ABThread(mod)
            self.ab_thread.result_ready.connect(
                lambda pos: continue_with_move(pos[0], pos[1]) if pos else continue_with_move(None, None))
            self.ab_thread.error_happened.connect(
                lambda e: continue_with_move(None, None, f"AB算法异常：{e}"))
            self.ab_thread.start()
        else:
            # 大模型算法（原本就是异步）
            status_now = 'playing'
            response_data = asyncio.run(
                send_yolo_result(pos_set, ai_pos_set, go_stones, status_now, mod))
            ai_down_pos_x, ai_down_pos_y, reason, mod = parase_response(response_data)
            continue_with_move(ai_down_pos_x, ai_down_pos_y, reason)



        # algorithm = self.algorithm_combo.currentText()
        # print('algorithm:',algorithm)
        # if algorithm == 'AB':
        #     machine_pos = AB_optimize.alpha_beta_process(mod)
        #     if not machine_pos:
        #         print("无法生成下一步走法，可能是棋局已结束！")
        #         ai_down_pos_x, ai_down_pos_y = 0, 0
        #     else:
        #         ai_down_pos_x, ai_down_pos_y = machine_pos
        # else:
        #     # 大模型算法
        #     status_now = 'playing'
        #     response_data = asyncio.run(
        #         send_yolo_result(pos_set, ai_pos_set, go_stones, status_now, mod))
        #     ai_down_pos_x, ai_down_pos_y, reason, mod = parase_response(response_data)

        # # 发送 YOLO 数据到 UDP（复用 game_analysis.py 逻辑）
        # yolo_data = {
        #     "black_pieces": list(ai_pos_set) if go_stones == "black" else list(pos_set),
        #     "white_pieces": list(pos_set) if go_stones == "black" else list(ai_pos_set),
        #     "ai_next_move": [ai_down_pos_x, ai_down_pos_y] if ai_down_pos_x is not None else None,
        #     "black_conf": list(ai_pos_set_conf) if go_stones == "black" else list(pos_set_conf),
        #     "white_conf": list(pos_set_conf) if go_stones == "black" else list(ai_pos_set_conf),
        # }
        # try:
        #     yolo_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #     yolo_socket.sendto(json.dumps(yolo_data).encode('utf-8'), (YOLO_UDP_IP, YOLO_UDP_PORT))
        #     print(f"发送 YOLO 数据: {yolo_data}")
        #     yolo_socket.close()
        # except socket.error as e:
        #     print(f"发送失败: {e}")
        #
        # # 更新历史列表显示 AI 走法
        # if ai_down_pos_x is not None:
        #     self.history_list.addItem(f"AI 推荐 ({ai_down_pos_x}, {ai_down_pos_y}) {ai_color} - 原因: {reason}")
        # else:
        #     self.history_list.addItem("无法生成下一步走法")

    def closeEvent(self, event):
        """关闭 socket (复用 qt_YOLO.py)"""
        self.yolo_socket.close()
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能五子棋数据收集系统 (集成 YOLO 和分析)")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("智能五子棋系统")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(title)

        # Tab widget for sections
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 添加摄像头管理模块
        self.camera_info = CameraInfoWindow()
        self.tab_widget.addTab(self.camera_info.centralWidget(), "摄像头管理模块")

        #图像处理模块
        self.image_process = TunerWindow()
        self.tab_widget.addTab(self.image_process.centralWidget(), "图像处理模块")

        # Tab 1: 数据收集 (原有)
        self.data_collection_page = DataCollectionPage()
        self.tab_widget.addTab(self.data_collection_page, "数据收集模块")

        # Tab 2: YOLO 棋局监控 (原有，集成分析)
        self.yolo_board = YOLOBoardWidget(self)
        self.tab_widget.addTab(self.yolo_board, "YOLO 棋局监控与分析")

        # Tab 3: 机械臂控制 (原有，复用 robot_arm.py 的 MainWindow)
        self.arm_control = ArmMainWindow()
        self.tab_widget.addTab(self.arm_control.centralWidget(), "机械臂控制模块")

        # Tab 4: 自动标注模块 (新增，复用 annotation_qt.py 的 AutoAnnotationGUI)
        self.auto_annotation = AutoAnnotationGUI()
        self.tab_widget.addTab(self.auto_annotation.centralWidget(), "自动标注模块")

        # 添加选项卡切换事件处理，以解决摄像头资源竞争
        self.tab_widget.currentChanged.connect(self.handle_tab_change)

        # Status bar
        self.statusBar().showMessage("系统就绪 (数据收集 + YOLO 监控 + 分析 + 机械臂控制 + 自动标注 + 摄像头管理)")

    def handle_tab_change(self, index):
        """处理选项卡切换：离开摄像头管理模块时释放摄像头资源"""
        camera_tab_index = 4  # 摄像头管理模块的固定索引（根据 addTab 顺序）
        if index != camera_tab_index:
            self.camera_info.stop_camera()
            print("已释放摄像头资源")  # 可选：调试日志

    def closeEvent(self, event):
        # 确保干净关闭 (包括 YOLO socket 和机械臂串口)
        self.yolo_board.closeEvent(event)
        if hasattr(self.arm_control, 'ser') and self.arm_control.ser is not None and self.arm_control.ser.is_open:
            self.arm_control.ser.close()
            # 确保摄像头管理模块正确关闭
        self.camera_info.closeEvent(event)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())