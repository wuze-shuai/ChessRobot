# detect_api.py
"""
封装官方 detect.py，保持 detect_self.py 同等接口
import cv2
from detect_api import YoloDetecter
detector = YoloDetecter(...)
img, boxes = detector.detect(im0)
"""
import sys, os, pathlib
ROOT = pathlib.Path(__file__).parents[0]  # yolov5 目录
sys.path.insert(0, str(ROOT))
import tempfile
import shutil
import cv2
import torch
from pathlib import Path
from detect import run, parse_opt  # 官方脚本同级目录


class YoloDetecter:
    """
    与 detect_self.py 完全对齐的调用方式
    """
    def __init__(self,
                 weights='yolov5s.pt',
                 data='data/coco128.yaml',
                 imgsz=(640, 640),
                 conf_thres=0.25,
                 iou_thres=0.45,
                 max_det=1000,
                 device='cpu',
                 classes=None,
                 agnostic_nms=False,
                 line_thickness=3,
                 hide_labels=False,
                 hide_conf=False,
                 half=False,
                 dnn=False):
        # 保存参数，detect 时再传
        self.weights = weights
        self.data = data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn

    def detect(self, im0):
        """
        :param im0: BGR numpy array (cv2 读入格式)
        :return: (out_img, boxes)
                 out_img: 画好框的 BGR 图
                 boxes:   [[x_center, y_center, w, h, conf, class], ...]  归一化坐标
        """
        # 1. 把输入图暂存到临时文件
        temp_dir = Path(tempfile.mkdtemp())
        temp_img = temp_dir / 'temp.jpg'
        cv2.imwrite(str(temp_img), im0)

        # 2. 输出目录
        out_dir = temp_dir / 'out'
        out_dir.mkdir()

        # 3. 拼参数 → 官方 run()
        opt = parse_opt()
        opt.weights = [self.weights]
        opt.source = str(temp_img)
        opt.data = self.data
        opt.imgsz = self.imgsz
        opt.conf_thres = self.conf_thres
        opt.iou_thres = self.iou_thres
        opt.max_det = self.max_det
        opt.device = self.device
        opt.classes = self.classes
        opt.agnostic_nms = self.agnostic_nms
        opt.line_thickness = self.line_thickness
        opt.hide_labels = self.hide_labels
        opt.hide_conf = self.hide_conf
        opt.half = self.half
        opt.dnn = self.dnn
        opt.project = str(out_dir)
        opt.name = ''
        opt.exist_ok = True
        opt.nosave = False
        opt.save_txt = True        # 必须保存 txt 才能拿到框
        opt.save_conf = True       # 同时保存置信度
        opt.save_format = 0        # YOLO 格式（归一化 xywh）
        opt.save_csv = False
        opt.save_crop = False
        opt.view_img = False
        opt.update = False
        opt.augment = False
        opt.visualize = False
        opt.vid_stride = 1

        # 4. 运行官方推理
        run(**vars(opt))

        # 5. 读取画框后的图
        out_img_path = out_dir / 'temp.jpg'
        out_img = cv2.imread(str(out_img_path))

        # 6. 读取归一化框
        boxes = []
        txt_file = out_dir / 'labels' / 'temp.txt'
        if txt_file.exists():
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # 无 conf
                        cls, x, y, w, h = map(float, parts)
                        conf = 1.0  # 默认 1
                    elif len(parts) == 6:  # 有 conf
                        cls, x, y, w, h, conf = map(float, parts)
                    else:
                        continue  # 异常行跳过
                    boxes.append([x, y, w, h, conf, int(cls)])

                    # *xywh, conf, cls = map(float, line.strip().split())
                    # boxes.append([*xywh, conf, int(cls)])

        # 7. 清理临时目录
        shutil.rmtree(temp_dir)

        return out_img, boxes


# 简单 CLI 测试
if __name__ == '__main__':
    import cv2
    det = YoloDetecter(weights='best.pt', device='cpu')
    img = cv2.imread(r'E:\13project\007ChessRobot\ChessRobot\yolov5\data\myData\images\val\test_video7_bright.jpg')
    out, bs = det.detect(img)
    print(bs)
    cv2.imshow('result', out)
    cv2.waitKey(0)