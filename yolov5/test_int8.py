# eval_yolo_precision.py
"""
单脚本评测 YoloDetecter 精度
python eval_yolo_precision.py
"""
import sys, os, json, pathlib, itertools
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import defaultdict
from detect_api import YoloDetecter  # 你的封装

# ---------------- 配置区 -----------------
WEIGHTS   = r'best.pt'          # 待测权重
IMG_DIR   = r'E:\13project\007ChessRobot\ChessRobot\yolov5\data\myData\images\val'
LBL_DIR   = r'E:\13project\007ChessRobot\ChessRobot\yolov5\data\myData\labels\val'
IMG_SUFFIX= ('.jpg', '.jpeg', '.png')
CONF_THRES= 0.001               # 推理时置信度阈值
IOU_THRES = 0.45                # NMS IoU 阈值
MAP_IOU_THR = np.arange(0.5, 1.0, 0.05)  # 0.5:0.05:0.95
# ----------------------------------------

def box_iou(box1, box2):
    """
    box1, box2: nx4  xyxy
    return: nxm IoU
    """
    def box_area(b):
        return (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    area1, area2 = box_area(box1), box_area(box2)
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    inter = (rb-lt).clamp(min=0).prod(dim=2)
    return inter / (area1[:, None] + area2 - inter)

def yolo2xyxy(b, img_w, img_h):
    """
    b: [x_center, y_center, w, h] 归一化
    return: xyxy 绝对像素
    """
    x, y, w, h = b[:, 0]*img_w, b[:, 1]*img_h, b[:, 2]*img_w, b[:, 3]*img_h
    x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
    return np.stack([x1, y1, x2, y2], axis=1)

def read_label(p: Path, img_w, img_h):
    """读取 YOLO 格式 GT"""
    if not p.exists():
        return [], []
    cls, xyxy = [], []
    with p.open() as f:
        for line in f:
            c, x, y, w, h = map(float, line.strip().split())
            cls.append(int(c))
            xyxy.append([x-w/2, y-h/2, x+w/2, y+h/2])
    if len(cls)==0:
        return np.zeros((0,5)), np.zeros((0,1))
    xyxy = np.array(xyxy) * [img_w, img_h, img_w, img_h]
    return xyxy, np.array(cls)[:, None]

def compute_ap(recall, precision):
    """11-point interpolated AP"""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1])

def evaluate():
    detector = YoloDetecter(weights=WEIGHTS,
                            conf_thres=CONF_THRES,
                            iou_thres=IOU_THRES,
                            device='cpu')
    img_paths = sorted([p for p in Path(IMG_DIR).iterdir()
                        if p.suffix.lower() in IMG_SUFFIX])
    assert img_paths, f'未找到图片，请检查 {IMG_DIR}'

    stats = []  # 保存 (tp, conf, pred_cls, target_cls)
    for img_p in img_paths:
        img = cv2.imread(str(img_p))
        H, W = img.shape[:2]

        # 1. 推理
        out_img, preds = detector.detect(img)  # preds: [[x,y,w,h,conf,cls], ...] 归一化
        if len(preds):
            preds = np.array(preds)
            pred_xyxy = yolo2xyxy(preds[:, :4], W, H)
            pred_conf = preds[:, 4]
            pred_cls  = preds[:, 5]
        else:
            pred_xyxy = np.zeros((0, 4))
            pred_conf = np.zeros(0)
            pred_cls  = np.zeros(0)

        # 2. GT
        lbl_p = Path(LBL_DIR) / (img_p.stem + '.txt')
        gt_xyxy, gt_cls = read_label(lbl_p, W, H)

        # 3. 匹配
        nl = len(gt_cls)
        correct = np.zeros((len(pred_xyxy), len(MAP_IOU_THR)), dtype=bool)
        if nl:
            detected = []
            for i, (pbox, pcls) in enumerate(zip(pred_xyxy, pred_cls)):
                ious = box_iou(torch.tensor(pbox[None]), torch.tensor(gt_xyxy)).numpy().ravel()
                for j, iou_thr in enumerate(MAP_IOU_THR):
                    # 先选同类且IoU满足且未被匹配的GT
                    idx = np.where((ious >= iou_thr) & (gt_cls.ravel() == pcls))[0]
                    idx = [k for k in idx if k not in detected]
                    if len(idx):
                        detected.append(idx[0])
                        correct[i, j] = True
                        break
        stats.append((correct, pred_conf, pred_cls, gt_cls))

        out_vis = out_img.copy()
        cv2.putText(out_vis, f'FP={len(pred_xyxy) - correct.sum()} TP={correct.sum()}',
                    (10, 30), 0, 0.8, (0, 0, 255), 2)
        save_path = Path('runs/val_vis') / img_p.name
        save_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(save_path), out_vis)

    # 4. 计算 mAP
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats)==0:
        print('未检测到任何目标，请检查权重或数据')
        return
    correct, conf, pred_cls, gt_cls = stats
    AP50, AP = [], []
    for iou_idx, iou_thr in enumerate(MAP_IOU_THR):
        sort_idx = np.argsort(-conf)
        correct_i = correct[sort_idx, iou_idx]
        pred_cls_i = pred_cls[sort_idx]
        tp = correct_i.sum()
        fp = (~correct_i).sum()
        fn = len(gt_cls) - tp
        precision = np.cumsum(correct_i) / (np.cumsum(correct_i) + np.cumsum(~correct_i))
        recall    = np.cumsum(correct_i) / len(gt_cls)
        ap = compute_ap(recall, precision)
        if abs(iou_thr - 0.5) < 1e-3:
            AP50.append(ap)
        AP.append(ap)
    print(f'mAP@0.5 : {np.mean(AP50):.4f}')
    print(f'mAP@0.5:0.95 : {np.mean(AP):.4f}')

    # 5. 计算 P/R/F1@0.5
    sort_idx = np.argsort(-conf)
    correct_50 = correct[sort_idx, 0]  # 0.5 列
    tp = correct_50.sum()
    fp = (~correct_50).sum()
    fn = len(gt_cls) - tp
    P = tp / (tp + fp + 1e-16)
    R = tp / (tp + fn + 1e-16)
    F1 = 2*P*R/(P+R+1e-16)
    print(f'Precision@0.5 : {P:.4f}')
    print(f'Recall@0.5    : {R:.4f}')
    print(f'F1@0.5        : {F1:.4f}')

if __name__ == '__main__':
    evaluate()