import sys
import os
import cv2
import time
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLineEdit, QLabel, QCheckBox, QFileDialog, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from image_find_focus import FocusFinder
import numpy as np

class CameraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摄像头捕获界面")
        self.setGeometry(100, 100, 800, 600)

        # 初始化变量
        self.cap = None
        self.capture_thread = None
        self.is_capturing = False
        self.focus_finder = FocusFinder()

        # 创建主窗口和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # 图像显示
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.image_label)

        # 控件布局
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        # 输出目录选择
        self.dir_input = QLineEdit('images')
        self.dir_button = QPushButton("选择输出目录")
        self.dir_button.clicked.connect(self.select_directory)
        controls_layout.addWidget(QLabel("输出目录："))
        controls_layout.addWidget(self.dir_input)
        controls_layout.addWidget(self.dir_button)

        # 时间间隔输入
        self.interval_input = QLineEdit('5')
        controls_layout.addWidget(QLabel("时间间隔（秒）："))
        controls_layout.addWidget(self.interval_input)

        # 模式选择
        self.mode_checkbox = QCheckBox("数据收集模式")
        controls_layout.addWidget(self.mode_checkbox)

        # 开始/停止按钮
        self.start_button = QPushButton("开始捕获")
        self.start_button.clicked.connect(self.start_capture)
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止捕获")
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        # 状态显示
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        main_layout.addWidget(self.status_text)

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.dir_input.setText(directory)

    def start_capture(self):
        if not self.is_capturing:
            try:
                interval = float(self.interval_input.text())
                output_dir = self.dir_input.text()
                get_data_mode = self.mode_checkbox.isChecked()

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                self.cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.status_text.append("错误：无法打开摄像头")
                    return

                self.is_capturing = True
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)

                self.capture_thread = threading.Thread(
                    target=self.capture_loop,
                    args=(interval, output_dir, get_data_mode)
                )
                self.capture_thread.daemon = True
                self.capture_thread.start()

                self.status_text.append(f"开始捕获到 {output_dir}")

            except ValueError:
                self.status_text.append("错误：无效的时间间隔值")

    def stop_capture(self):
        if self.is_capturing:
            self.is_capturing = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            if self.cap:
                self.cap.release()
                self.cap = None
            self.status_text.append("捕获已停止，摄像头已释放")

    def capture_loop(self, interval, output_dir, get_data_mode):
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                self.status_text.append("错误：无法捕获帧")
                continue

            focus_image, has_res = self.focus_finder.find_focus(frame)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if get_data_mode:
                filename = os.path.join(output_dir, f"{timestamp}.jpg")
            else:
                interval = 1
                filename = os.path.join(output_dir, f"0001testimage.jpg")
            filename_ori = os.path.join(output_dir, f"0001originalimage.jpg")

            cv2.imwrite(filename_ori, frame)
            cv2.imwrite(filename, focus_image)

            # 更新图像显示
            rgb_image = cv2.cvtColor(focus_image, cv2.COLOR_BGR2RGB)
            try:
                rgb_image = focus_image.astype(np.uint8)
            except Exception as e:
                print(f"未读取到完整棋盘: {e}")
                
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)

            self.status_text.append(f"图片已保存：{filename}")

            time.sleep(interval)

    def closeEvent(self, event):
        self.stop_capture()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraGUI()
    window.show()
    sys.exit(app.exec_())