import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QListWidget, QLabel, QPushButton, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QCameraInfo, QCamera
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtGui import QFont


class CameraInfoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摄像头信息查看器")
        self.setGeometry(100, 100, 800, 600)

        # 当前活动的摄像头对象
        self.current_camera = None
        # 存储摄像头信息和对应的 cv2 索引
        self.camera_indices = {}

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Camera list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Title
        title = QLabel("可用摄像头")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        # Camera list
        self.camera_list = QListWidget()
        self.camera_list.itemClicked.connect(self.display_camera_info)
        left_layout.addWidget(self.camera_list)

        # Refresh button
        refresh_button = QPushButton("刷新摄像头列表")
        refresh_button.clicked.connect(self.populate_camera_list)
        left_layout.addWidget(refresh_button)

        # Right panel: Camera details and video
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Details title
        details_title = QLabel("摄像头信息")
        details_title.setFont(QFont("Arial", 14, QFont.Bold))
        details_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(details_title)

        # Details display
        self.details_display = QTextEdit()
        self.details_display.setReadOnly(True)
        self.details_display.setFixedHeight(150)
        right_layout.addWidget(self.details_display)

        # Video display
        video_title = QLabel("摄像头画面")
        video_title.setFont(QFont("Arial", 14, QFont.Bold))
        video_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(video_title)

        self.viewfinder = QCameraViewfinder()
        self.viewfinder.setMinimumHeight(300)
        right_layout.addWidget(self.viewfinder)
        right_layout.addStretch()

        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)

        # Populate camera list initially
        self.populate_camera_list()

        # Status bar
        self.statusBar().showMessage("系统就绪")

    def populate_camera_list(self):
        """Populate the list with available cameras and assign cv2 indices"""
        self.camera_list.clear()
        self.camera_indices.clear()
        cameras = QCameraInfo.availableCameras()
        if not cameras:
            self.camera_list.addItem("未检测到摄像头")
            self.details_display.setText("未找到可用摄像头。")
            self.statusBar().showMessage("未检测到摄像头")
            self.stop_camera()
            return

        for index, camera in enumerate(cameras):
            name = camera.description() or camera.deviceName()[:30] or f"Camera {index}"
            self.camera_list.addItem(name)
            self.camera_indices[name] = index  # 存储名称到索引的映射
            print(f"检测到摄像头: {name}, 描述: {camera.description() or '无描述'}, cv2 索引: {index}")
        self.statusBar().showMessage(f"检测到 {len(cameras)} 个摄像头")

    def display_camera_info(self, item):
        """Display detailed information and video feed of the selected camera"""
        try:
            # Stop any currently running camera
            self.stop_camera()

            cameras = QCameraInfo.availableCameras()
            if not cameras:
                self.details_display.setText("未找到可用摄像头")
                self.statusBar().showMessage("未找到可用摄像头")
                return

            selected_name = item.text()
            camera_info = None
            for camera in cameras:
                if (camera.description() == selected_name or
                        camera.deviceName()[:30] == selected_name or
                        selected_name == f"Camera {cameras.index(camera)}"):
                    camera_info = camera
                    break

            if camera_info:
                # Display camera details
                details = f"设备名称: {camera_info.deviceName() or '未知'}\n"
                details += f"描述: {camera_info.description() or '无描述'}\n"
                details += f"位置: {self.get_position_string(camera_info.position())}\n"
                details += f"方向: {camera_info.orientation()}°\n"
                details += f"cv2 索引: {self.camera_indices.get(selected_name, '未知')}\n"
                details += "支持的分辨率:\n"
                try:
                    # Try to get resolutions via QCamera
                    self.current_camera = QCamera(camera_info)
                    self.current_camera.load()
                    viewfinder_settings = self.current_camera.supportedViewfinderSettings()
                    if viewfinder_settings:
                        for setting in viewfinder_settings:
                            resolution = setting.resolution()
                            details += f"  {resolution.width()}x{resolution.height()}\n"
                    else:
                        details += "  无可用分辨率信息\n"
                    # Start camera and bind to viewfinder
                    self.current_camera.setViewfinder(self.viewfinder)
                    self.current_camera.start()
                except Exception as res_error:
                    details += f"  获取分辨率失败: {str(res_error)}\n"
                    self.current_camera = None
                self.details_display.setText(details)
                self.statusBar().showMessage(f"显示 {camera_info.deviceName() or '未知'} 的信息")
            else:
                self.details_display.setText("无法获取摄像头信息")
                self.statusBar().showMessage("未选择有效摄像头")
        except Exception as e:
            self.details_display.setText(f"错误: {str(e)}")
            self.statusBar().showMessage(f"获取摄像头信息失败: {str(e)}")
            print(f"错误: {str(e)}")

    def stop_camera(self):
        """Stop and release the current camera"""
        if self.current_camera:
            self.current_camera.stop()
            self.current_camera.unload()
            self.current_camera = None

    def get_position_string(self, position):
        """Convert camera position enum to readable string"""
        try:
            if position == QCamera.Position.FrontFace:
                return "前置摄像头"
            elif position == QCamera.Position.BackFace:
                return "后置摄像头"
            elif position == QCamera.Position.UnspecifiedPosition:
                return "位置未知"
            return "未知"
        except AttributeError as e:
            print(f"位置枚举错误: {str(e)}")
            return "位置信息不可用"

    def closeEvent(self, event):
        """Ensure camera is stopped when closing the window"""
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraInfoWindow()
    window.show()
    sys.exit(app.exec_())