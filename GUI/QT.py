# -*- coding: utf-8 -*-
# QT.py (optimized with data collection module from camera_data_qt.py)
# This version integrates the data collection module by importing camera_data_qt.py.

import sys
import json
import time
import socket
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QListWidget, QStackedWidget, QLabel, QPushButton, QComboBox,
                             QLineEdit, QTextEdit, QFrame, QGridLayout, QTabWidget, QProgressBar)
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize

import sys
import os
# 将dir2的路径添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import qt_main.py as a module
import qt_main
import camera_data_qt

# UDP settings from qt_main.py
YOLO_UDP_IP = qt_main.YOLO_UDP_IP
YOLO_UDP_PORT = qt_main.YOLO_UDP_PORT

# Chessboard settings (15x15 Gobang)
BOARD_SIZE = 15
GRID_SIZE = 40
BOARD_WIDTH = BOARD_SIZE * GRID_SIZE
BOARD_HEIGHT = BOARD_SIZE * GRID_SIZE
MARGIN = 20

class UDPListenerThread(QThread):
    data_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((YOLO_UDP_IP, YOLO_UDP_PORT))
        self.running = True

    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                yolo_data = json.loads(data.decode('utf-8'))
                self.data_received.emit(yolo_data)
            except Exception as e:
                print(f"UDP receive error: {e}")

    def stop(self):
        self.running = False
        self.sock.close()

class ChessBoardWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(BOARD_WIDTH + 2 * MARGIN, BOARD_HEIGHT + 2 * MARGIN)
        self.black_pieces = {}
        self.white_pieces = {}
        self.black_conf = []
        self.white_conf = []

    def update_pieces(self, yolo_data):
        self.black_pieces.clear()
        self.white_pieces.clear()
        for pos in yolo_data.get('black_pieces', []):
            x, y = pos
            conf = next((c for px, py, c in yolo_data.get('black_conf', []) if px == x and py == y), 1.0)
            self.black_pieces[(int(x), int(y))] = conf
        for pos in yolo_data.get('white_pieces', []):
            x, y = pos
            conf = next((c for px, py, c in yolo_data.get('white_conf', []) if px == x and py == y), 1.0)
            self.white_pieces[(int(x), int(y))] = conf
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(245, 222, 179))  # Light wood color for board

        # Draw grid
        pen = QPen(QColor(139, 69, 19), 2)
        painter.setPen(pen)
        for i in range(BOARD_SIZE + 1):
            painter.drawLine(MARGIN, MARGIN + i * GRID_SIZE, MARGIN + BOARD_WIDTH, MARGIN + i * GRID_SIZE)
            painter.drawLine(MARGIN + i * GRID_SIZE, MARGIN, MARGIN + i * GRID_SIZE, MARGIN + BOARD_HEIGHT)

        # Draw pieces
        for (x, y), conf in self.black_pieces.items():
            cx = MARGIN + x * GRID_SIZE + GRID_SIZE // 2
            cy = MARGIN + y * GRID_SIZE + GRID_SIZE // 2
            painter.setBrush(QBrush(QColor(0, 0, 0)))
            painter.drawEllipse(cx - GRID_SIZE // 3, cy - GRID_SIZE // 3, 2 * GRID_SIZE // 3, 2 * GRID_SIZE // 3)
            if conf < 1.0:
                painter.setFont(QFont("Arial", 8))
                painter.setPen(QColor(255, 0, 0))
                painter.drawText(cx + 10, cy + 5, f"{conf:.2f}")

        for (x, y), conf in self.white_pieces.items():
            cx = MARGIN + x * GRID_SIZE + GRID_SIZE // 2
            cy = MARGIN + y * GRID_SIZE + GRID_SIZE // 2
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawEllipse(cx - GRID_SIZE // 3, cy - GRID_SIZE // 3, 2 * GRID_SIZE // 3, 2 * GRID_SIZE // 3)
            if conf < 1.0:
                painter.setFont(QFont("Arial", 8))
                painter.setPen(QColor(255, 0, 0))
                painter.drawText(cx + 10, cy + 5, f"{conf:.2f}")

class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("欢迎使用智能五子棋机械臂控制系统")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        desc = QLabel("这是一个集成YOLO检测、AI决策和机械臂控制的五子棋对弈系统。\n请选择左侧导航进入相应模块。")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)
        layout.addStretch()

class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("系统设置")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        grid = QGridLayout()
        self.ip_edit = QLineEdit("127.0.0.1")
        self.port_edit = QLineEdit("5005")
        self.mod_combo = QComboBox()
        self.mod_combo.addItems(["简单", "中等", "固若金汤", "和我一样6的Level"])
        grid.addWidget(QLabel("UDP IP:"), 0, 0)
        grid.addWidget(self.ip_edit, 0, 1)
        grid.addWidget(QLabel("UDP Port:"), 1, 0)
        grid.addWidget(self.port_edit, 1, 1)
        grid.addWidget(QLabel("AI难度:"), 2, 0)
        grid.addWidget(self.mod_combo, 2, 1)
        layout.addLayout(grid)

        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        layout.addStretch()

    def save_settings(self):
        global qt_main
        qt_main.mod = self.mod_combo.currentText()
        print(f"设置保存: AI难度 = {qt_main.mod}")

class ArmControlPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        title = QLabel("机械臂控制")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        control_layout = QVBoxLayout()
        self.start_btn = QPushButton("开始控制")
        self.start_btn.clicked.connect(self.start_control)
        self.stop_btn = QPushButton("停止控制")
        self.stop_btn.clicked.connect(self.stop_control)
        self.calibrate_btn = QPushButton("校准机械臂")
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.calibrate_btn)
        control_layout.addStretch()

        self.chessboard = ChessBoardWidget()
        layout.addLayout(control_layout, 1)
        layout.addWidget(self.chessboard, 3)

        self.detect_timer = QTimer(self)
        self.detect_timer.timeout.connect(self.run_qt_main_detection)

    def start_control(self):
        self.detect_timer.start(1000)
        print("机械臂控制启动")
        qt_main.start_qt_app()

    def stop_control(self):
        self.detect_timer.stop()
        global qt_main
        qt_main.detect_flag = False
        print("机械臂控制停止")

    def run_qt_main_detection(self):
        global qt_main
        qt_main.detct(qt_main.pre_img, qt_main.self_yolo, qt_main.mod, qt_main.go_stones, qt_main.status_now)
        self.chessboard.update_pieces({"black_pieces": qt_main.history_set, "white_pieces": set()})

class ImageProcessPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("图像处理")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        self.image_label = QLabel("图像显示区域")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid black; background-color: gray;")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        detect_btn = QPushButton("开始检测")
        detect_btn.clicked.connect(self.start_detection)
        layout.addWidget(detect_btn)
        layout.addStretch()

    def start_detection(self):
        print("开始图像检测")
        pixmap = QPixmap("res_img.jpg")
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

class AIPlayPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("AI对弈")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        steps_layout = QHBoxLayout()
        self.step1_btn = QPushButton("步骤1: 初始化")
        self.step2_btn = QPushButton("步骤2: 检测")
        self.step3_btn = QPushButton("步骤3: 下棋")
        self.step4_btn = QPushButton("步骤4: 分析")
        self.step5_btn = QPushButton("步骤5: 结束")
        steps_layout.addWidget(self.step1_btn)
        steps_layout.addWidget(self.step2_btn)
        steps_layout.addWidget(self.step3_btn)
        steps_layout.addWidget(self.step4_btn)
        steps_layout.addWidget(self.step5_btn)
        layout.addLayout(steps_layout)

        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_label = QLabel("请选择步骤以查看详情")
        self.content_layout.addWidget(self.content_label)
        layout.addWidget(self.content_frame)

        options_layout = QVBoxLayout()
        self.color_combo = QComboBox()
        self.color_combo.addItems(["黑子", "白子"])
        options_layout.addWidget(QLabel("执棋颜色:"))
        options_layout.addWidget(self.color_combo)
        self.color_combo.currentTextChanged.connect(self.update_color)
        self.status_label = QLabel("状态: 等待")
        options_layout.addWidget(self.status_label)
        play_btn = QPushButton("开始对弈")
        play_btn.clicked.connect(self.start_play)
        options_layout.addWidget(play_btn)
        options_layout.addStretch()
        layout.addLayout(options_layout)

        self.step1_btn.clicked.connect(lambda: self.update_step_content("初始化系统和机械臂"))
        self.step2_btn.clicked.connect(lambda: self.update_step_content("进行YOLO图像检测"))
        self.step3_btn.clicked.connect(lambda: self.update_step_content("AI决策并机械臂下棋"))
        self.step4_btn.clicked.connect(lambda: self.update_step_content("分析棋局和胜负"))
        self.step5_btn.clicked.connect(lambda: self.update_step_content("结束对弈并清理"))

    def update_step_content(self, text):
        self.content_label.setText(text)

    def update_color(self, color):
        global qt_main
        qt_main.go_stones = "black" if color == "黑子" else "white"
        print(f"更新执棋颜色: {qt_main.go_stones}")

    def start_play(self):
        global qt_main
        qt_main.status_now = 'playing'
        print("开始AI对弈")

class DataAnalysisPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("数据分析")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        self.log_text = QTextEdit()
        self.log_text.setPlaceholderText("日志和分析数据将显示在这里...")
        layout.addWidget(self.log_text)
        layout.addStretch()

class HelpPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("帮助")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        help_text = QLabel("用户手册和常见问题解答。\n联系支持：support@example.com")
        help_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(help_text)
        layout.addStretch()

class DataCollectionPage(camera_data_qt.CameraGUI):
    def __init__(self):
        super().__init__()
        # Customize the title or other properties if needed
        self.setWindowTitle("数据收集模块")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能五子棋机械臂控制系统")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left navigation sidebar
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(150)
        nav_items = ["首页", "系统设置", "机械臂控制", "图像处理", "AI对弈", "数据分析", "帮助", "数据收集"]
        self.nav_list.addItems(nav_items)
        self.nav_list.itemClicked.connect(self.switch_page)
        main_layout.addWidget(self.nav_list)

        # Central stacked widget for pages
        self.stacked_widget = QStackedWidget()
        self.pages = [
            HomePage(),
            SettingsPage(),
            ArmControlPage(),
            ImageProcessPage(),
            AIPlayPage(),
            DataAnalysisPage(),
            HelpPage(),
            DataCollectionPage()
        ]
        for page in self.pages:
            self.stacked_widget.addWidget(page)
        main_layout.addWidget(self.stacked_widget)

        # Status bar
        self.statusBar().showMessage("系统就绪")

        # UDP listener
        self.udp_thread = UDPListenerThread()
        self.udp_thread.data_received.connect(self.on_data_received)
        self.udp_thread.start()

        # Initial page
        self.nav_list.setCurrentRow(0)
        self.switch_page(self.nav_list.item(0))

    def switch_page(self, item):
        index = self.nav_list.row(item)
        self.stacked_widget.setCurrentIndex(index)

    def on_data_received(self, yolo_data):
        if len(self.pages) > 2:
            if isinstance(self.pages[2], ArmControlPage):
                self.pages[2].chessboard.update_pieces(yolo_data)
        self.statusBar().showMessage(f"收到YOLO数据: {len(yolo_data.get('black_pieces', []))} 黑子, {len(yolo_data.get('white_pieces', []))} 白子")

    def closeEvent(self, event):
        global qt_main
        qt_main.detect_flag = False
        self.udp_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())