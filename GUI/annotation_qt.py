import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QDoubleSpinBox,
    QSpinBox, QComboBox, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
from pathlib import Path
# 添加 YOLOv5 目录到 sys.path
yolov5_dir = Path(r"E:\13project\007ChessRobot\ChessRobot\yolov5")
sys.path.append(str(yolov5_dir))
from yolov5.annotation import run_yolo_detection, \
    load_classes_from_yaml  # Assuming the provided script is saved as annotation.py

def key_to_chinese(key):
    """将参数名翻译为中文"""
    translations = {
        'image_dir': '图像目录',
        'save_dir': '保存目录',
        'weights': '权重文件',
        'data': '数据 YAML 文件'
    }
    return translations.get(key, key.replace('_', ' ').title())

class DetectionThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            # Load classes from YAML
            classes_path = Path(self.params['save_dir']) / 'classes.txt'
            load_classes_from_yaml(self.params['data'], classes_path)

            # Get image files for progress tracking
            image_dir = Path(self.params['image_dir'])
            image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
            total_images = len(image_files)

            # Monkey-patch the run_yolo_detection to emit progress
            original_print = print

            def progress_print(msg):
                if "无法读取图像" in msg:
                    original_print(msg)
                elif "检测结果已保存到" in msg:
                    self.finished.emit(msg)
                else:
                    original_print(msg)

            # We can't directly patch the loop, but for simplicity, we'll assume progress per image
            # In actual run_yolo_detection, add progress emission if modifiable, but here simulate
            run_yolo_detection(**self.params)  # Run the detection

            # Simulate progress if not patched
            for i in range(total_images):
                self.progress.emit(int((i + 1) / total_images * 100))
                # In real, this would be inside the loop of run_yolo_detection

            self.finished.emit("检测完成成功。")
        except Exception as e:
            self.error.emit(str(e))

class AutoAnnotationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 自动标注工具")
        self.setGeometry(100, 100, 600, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # 图像目录
        self.add_file_input("图像目录：", "image_dir_line", "选择图像目录", is_dir=True)

        # 保存目录
        self.add_file_input("保存目录：", "save_dir_line", "选择保存目录", is_dir=True)

        # 权重文件
        self.add_file_input("权重文件：", "weights_line", "选择权重文件", is_dir=False, file_filter="*.pt")

        # 数据 YAML 文件
        self.add_file_input("数据 YAML 文件：", "data_line", "选择数据 YAML 文件", is_dir=False, file_filter="*.yaml")

        # 图像尺寸
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1280)
        self.img_size_spin.setValue(640)
        self.img_size_spin.setSingleStep(32)
        self.add_labeled_widget("图像尺寸：", self.img_size_spin)

        # 置信度阈值
        self.conf_thres_spin = QDoubleSpinBox()
        self.conf_thres_spin.setRange(0.0, 1.0)
        self.conf_thres_spin.setValue(0.25)
        self.conf_thres_spin.setSingleStep(0.01)
        self.add_labeled_widget("置信度阈值：", self.conf_thres_spin)

        # IOU 阈值
        self.iou_thres_spin = QDoubleSpinBox()
        self.iou_thres_spin.setRange(0.0, 1.0)
        self.iou_thres_spin.setValue(0.45)
        self.iou_thres_spin.setSingleStep(0.01)
        self.add_labeled_widget("IOU 阈值：", self.iou_thres_spin)

        # 设备
        self.device_combo = QComboBox()
        self.device_combo.addItems(["", "0", "cpu"])  # Add more options if needed
        self.add_labeled_widget("设备：", self.device_combo)

        # 开始按钮
        self.start_button = QPushButton("开始自动标注")
        self.start_button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.start_button)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

    def add_labeled_widget(self, label_text, widget):
        h_layout = QHBoxLayout()
        label = QLabel(label_text)
        h_layout.addWidget(label)
        h_layout.addWidget(widget)
        self.layout.addLayout(h_layout)

    def add_file_input(self, label_text, line_attr, button_text, is_dir, file_filter=None):
        h_layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
        button = QPushButton(button_text)

        if is_dir:
            button.clicked.connect(lambda: self.select_directory(line_edit))
        else:
            button.clicked.connect(lambda: self.select_file(line_edit, file_filter))

        h_layout.addWidget(label)
        h_layout.addWidget(line_edit)
        h_layout.addWidget(button)
        self.layout.addLayout(h_layout)
        setattr(self, line_attr, line_edit)

    def select_directory(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "选择目录")
        if dir_path:
            line_edit.setText(dir_path)

    def select_file(self, line_edit, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter or "*.*")
        if file_path:
            line_edit.setText(file_path)

    def start_detection(self):
        params = {
            'image_dir': self.image_dir_line.text(),
            'save_dir': self.save_dir_line.text(),
            'weights': self.weights_line.text(),
            'data': self.data_line.text(),
            'img_size': self.img_size_spin.value(),
            'conf_thres': self.conf_thres_spin.value(),
            'iou_thres': self.iou_thres_spin.value(),
            'device': self.device_combo.currentText()
        }

        # 验证输入
        for key, value in params.items():
            if key in ['image_dir', 'save_dir', 'weights', 'data'] and not value:
                QMessageBox.warning(self, "输入错误", f"请提供 {key_to_chinese(key)}。")
                return
            if key in ['image_dir', 'save_dir'] and not os.path.isdir(value):
                QMessageBox.warning(self, "输入错误", f"{key_to_chinese(key)} 不是有效目录。")
                return
            if key in ['weights', 'data'] and not os.path.isfile(value):
                QMessageBox.warning(self, "输入错误", f"{key_to_chinese(key)} 不是有效文件。")
                return

        self.start_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.thread = DetectionThread(params)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_finished(self, message):
        self.start_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "成功", message)

    def on_error(self, error):
        self.start_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "错误", f"发生错误：{error}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoAnnotationGUI()
    window.show()
    sys.exit(app.exec_())