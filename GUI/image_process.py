# image_process.py  (去工具栏版)
import sys, cv2, numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
        QPushButton, QSlider, QSpinBox, QHBoxLayout, QVBoxLayout,
        QFileDialog, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from Tools.image_find_focus import FocusFinder


IMG_W, IMG_H = 320, 240
STEP_DESC = ["1. 灰度图", "2. HSV 掩膜", "3. 形态学处理", "4. Canny 边缘",
             "5. 最大轮廓/角点", "6. 透视矫正结果"]

class Cv2ToQt:
    @staticmethod
    def to_pixmap(cv_bgr):
        if cv_bgr is None or cv_bgr.size == 0:
            return QPixmap()
        rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(IMG_W, IMG_H, Qt.KeepAspectRatio)

class TunerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理参数调试器（全中文）")
        self.resize(1200, 700)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.slot_continuous)
        self.src_bgr = None
        self.focus_finder = FocusFinder()

        central = QWidget()
        self.setCentralWidget(central)
        master = QHBoxLayout(central)

        # 左侧：参数面板 + 按钮区
        left = QVBoxLayout()
        left.addLayout(self.build_button_bar())   # ← 原来工具栏的按钮放这里
        left.addLayout(self.build_param_panel())
        master.addLayout(left, stretch=1)

        # 右侧：6 张图
        self.lbl_steps = [QLabel(STEP_DESC[i]) for i in range(6)]
        for l in self.lbl_steps:
            l.setFixedSize(IMG_W, IMG_H)
            l.setStyleSheet("border:1px solid gray;")
        grid = QVBoxLayout()
        for i in range(0, 6, 2):
            h = QHBoxLayout()
            h.addWidget(self.lbl_steps[i])
            h.addWidget(self.lbl_steps[i+1])
            grid.addLayout(h)
        master.addLayout(grid, stretch=2)

    # ---------- 原工具栏按钮 → 横向按钮条 ----------
    def build_button_bar(self):
        h = QHBoxLayout()
        h.addWidget(QPushButton("打开图片", clicked=self.slot_open_img))
        h.addWidget(QPushButton("打开摄像头", clicked=self.slot_choose_cam))
        h.addWidget(QPushButton("关闭摄像头", clicked=self.slot_close_cam))

        # 手动分隔符
        sep = QLabel(" | ")
        sep.setStyleSheet("color:gray;")
        h.addWidget(sep)

        h.addWidget(QPushButton("单步运行", clicked=self.slot_single_step))
        h.addWidget(QPushButton("连续运行", clicked=self.slot_continuous))
        h.addWidget(QPushButton("停止连续", clicked=self.timer.stop))
        h.addStretch()
        return h

    # ---------- 参数面板（未改动） ----------
    def build_param_panel(self):
        v = QVBoxLayout()
        v.addWidget(QLabel("<h3>可调参数（已汉化）</h3>"))

        self.cb_cam_id = QComboBox()
        self.cb_cam_id.addItems([str(i) for i in range(4)])
        v.addWidget(QLabel("摄像头编号（0/1/2/3）"))
        v.addWidget(self.cb_cam_id)

        self.sb_interval = QSpinBox()
        self.sb_interval.setRange(30, 5000)
        self.sb_interval.setValue(200)
        self.sb_interval.setSuffix(" ms")
        v.addWidget(QLabel("连续运行间隔（30-5000 ms）"))
        v.addWidget(self.sb_interval)

        self.sld_dark = QSlider(Qt.Horizontal)
        self.sld_dark.setRange(0, 150)
        self.sld_dark.setValue(0)
        v.addWidget(QLabel("HSV掩膜亮度补偿（0-150，越大越易检出暗区）"))
        v.addWidget(self.sld_dark)

        self.sp_hmax = QSpinBox()
        self.sp_hmax.setRange(0, 180)
        self.sp_hmax.setValue(45)
        v.addWidget(QLabel("HSV色调上限（0-180，过滤色偏）"))
        v.addWidget(self.sp_hmax)

        self.sp_morph = QSpinBox()
        self.sp_morph.setRange(3, 19)
        self.sp_morph.setSingleStep(2)
        self.sp_morph.setValue(5)
        v.addWidget(QLabel("形态学核大小（3-19，奇数，去噪用）"))
        v.addWidget(self.sp_morph)

        self.sp_canny_low = QSpinBox()
        self.sp_canny_low.setRange(10, 250)
        self.sp_canny_low.setValue(30)
        v.addWidget(QLabel("Canny低阈值（10-250，边缘灵敏度）"))
        v.addWidget(self.sp_canny_low)

        self.sp_canny_high = QSpinBox()
        self.sp_canny_high.setRange(50, 300)
        self.sp_canny_high.setValue(250)
        v.addWidget(QLabel("Canny高阈值（50-300，抑制假边缘）"))
        v.addWidget(self.sp_canny_high)

        self.sp_eps = QSpinBox()
        self.sp_eps.setRange(10, 200)
        self.sp_eps.setValue(50)
        v.addWidget(QLabel("轮廓近似epsilon（10-200，越大折线越简）"))
        v.addWidget(self.sp_eps)

        self.sp_quality = QSpinBox()
        self.sp_quality.setRange(1, 100)
        self.sp_quality.setValue(60)
        v.addWidget(QLabel("角点质量百分下限（1-100，越高越严格）"))
        v.addWidget(self.sp_quality)

        self.sp_minDist = QSpinBox()
        self.sp_minDist.setRange(50, 500)
        self.sp_minDist.setValue(200)
        v.addWidget(QLabel("角点最小间距（50-500，像素）"))
        v.addWidget(self.sp_minDist)

        v.addStretch()
        return v

    # ---------- 以下所有槽函数/算法均未改动 ----------
    def slot_open_img(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片(*.png *.jpg *.bmp)")
        if path:
            self.src_bgr = cv2.imread(path)
            if self.src_bgr is None:
                QMessageBox.warning(self, "错误", "无法读取图片")
            else:
                self.slot_single_step()

    def slot_choose_cam(self):
        self.slot_close_cam()
        cam_id = int(self.cb_cam_id.currentText())
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开摄像头 {cam_id}")
            self.cap = None
        else:
            self.slot_continuous()

    def slot_close_cam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def slot_single_step(self):
        if self.src_bgr is None:
            QMessageBox.information(self, "提示", "请先打开图片或摄像头")
            return
        self.process_and_show(self.src_bgr)

    def slot_continuous(self):
        if self.cap and self.cap.isOpened():
            self.timer.setInterval(self.sb_interval.value())
            ret, frame = self.cap.read()
            if ret:
                self.src_bgr = frame.copy()
                self.process_and_show(frame)
            else:
                self.slot_close_cam()
        else:
            self.timer.stop()

    # ---------- 核心处理 ----------
    def process_and_show(self, bgr):
        # 读参
        darkness = self.sld_dark.value()
        hmax = self.sp_hmax.value()
        k_size = self.sp_morph.value()
        canny_low = self.sp_canny_low.value()
        canny_high = self.sp_canny_high.value()
        eps = self.sp_eps.value()
        quality = self.sp_quality.value() / 100.0
        min_dist = self.sp_minDist.value()

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mean_b = int(np.mean(gray) * 150 / 255)
        self.lbl_steps[0].setPixmap(Cv2ToQt.to_pixmap(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, darkness + mean_b])
        upper = np.array([hmax, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        self.lbl_steps[1].setPixmap(Cv2ToQt.to_pixmap(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))

        kernel = np.ones((k_size, k_size), np.uint8)
        close = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1),
                                  cv2.MORPH_CLOSE, kernel, iterations=1)
        self.lbl_steps[2].setPixmap(Cv2ToQt.to_pixmap(cv2.cvtColor(close, cv2.COLOR_GRAY2BGR)))

        canny = cv2.Canny(close, canny_low, canny_high)
        canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        self.lbl_steps[3].setPixmap(Cv2ToQt.to_pixmap(cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)))

        cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.arcLength(cnt, True) >= 1000:
                poly = cv2.approxPolyDP(cnt, eps, True)
                canvas = bgr.copy()
                cv2.drawContours(canvas, [poly], -1, (0, 255, 0), 2)
                gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray_canvas, 4, quality, min_dist)
                if corners is not None:
                    for c in np.intp(corners):
                        x, y = c.ravel()
                        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
                self.lbl_steps[4].setPixmap(Cv2ToQt.to_pixmap(canvas))
            else:
                self.lbl_steps[4].setText("最大轮廓长度不足")
        else:
            self.lbl_steps[4].setText("未检出轮廓")

        warp, ok = self.my_perspective(bgr, darkness, hmax, k_size, canny_low, canny_high, eps, quality, min_dist)
        if ok and warp is not None:
            self.lbl_steps[5].setPixmap(Cv2ToQt.to_pixmap(warp))
        else:
            self.lbl_steps[5].setText("透视失败")

    # ---------- 透视 ----------
    def my_perspective(self, bgr, darkness, hmax, k_size, canny_low, canny_high, eps, quality, min_dist):
        finder = self.focus_finder
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mean_b = int(np.mean(gray) * 150 / 255)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, darkness + mean_b])
        upper = np.array([hmax, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((k_size, k_size), np.uint8)
        close = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1),
                                  cv2.MORPH_CLOSE, kernel, iterations=1)
        canny = cv2.Canny(close, canny_low, canny_high)
        canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, False
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.arcLength(cnt, True) < 1000:
            return None, False
        poly = cv2.approxPolyDP(cnt, eps, True)
        temp_canvas = np.ones(canny.shape, np.uint8) * 255
        cv2.drawContours(temp_canvas, [poly], -1, 0, 1)
        corners = cv2.goodFeaturesToTrack(temp_canvas, 4, quality, min_dist)
        if corners is None:
            return None, False
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(temp_canvas, corners, (11, 11), (-1, -1), criteria)
        corners = np.intp(corners)
        pts = [tuple(c.ravel()) for c in corners]
        rect = finder.sort_corner(finder.find_corner(pts))
        dst = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
        M = cv2.getPerspectiveTransform(np.float32(rect), dst)
        warp = cv2.warpPerspective(bgr, M, (500, 500))
        warp = cv2.flip(warp, 1)
        return warp, True

# -------------------- main --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TunerWindow()
    w.show()
    sys.exit(app.exec_())