# evaluate_model.py
"""
This script evaluates the performance of the 'best.pt' model on a YOLOv5 dataset.
It uses the YoloDetecter from detect_api.py to perform inference on validation images,
parses ground truth labels, and computes metrics like Precision, Recall, AP, and mAP.

Assumptions:
- Dataset is in YOLO format: images/val/ and labels/val/ under the dataset root.
- Labels are in YOLO format: class x_center y_center width height (normalized).
- Model outputs are in the same format.
- For multi-class, computes AP per class and mAP.
- IoU threshold for matching is 0.5 (configurable).
- Uses non-max suppression implicitly via the detector.

Requirements:
- Place this script in the same directory as detect_api.py.
- Adjust DATASET_ROOT, IMG_DIR, LABEL_DIR as needed.
- Install necessary libraries if not already (numpy, etc.), but assumes YOLOv5 env.

Usage:
python evaluate_model.py
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from detect_api import YoloDetecter  # Your custom detector wrapper

# Configuration
WEIGHTS = 'best.onnx'  # Path to your model weights
DATA = 'data/myData.yaml'
DATASET_ROOT = 'data/myData/'  # Adjust to your dataset root, e.g., 'data/myData/'
IMG_DIR = os.path.join(DATASET_ROOT, 'images/train/')  # Validation images
LABEL_DIR = os.path.join(DATASET_ROOT, 'labels/train/')  # Ground truth labels
IMG_SIZE = (640, 640)  # Model input size
CONF_THRES = 0.25  # Confidence threshold
IOU_THRES = 0.45  # NMS IoU threshold
EVAL_IOU_THRES = 0.5  # IoU threshold for TP in evaluation
DEVICE = 'cpu'  # 'cuda' if available
CLASSES = None  # List of class indices if filtering, else None for all

# If you have a yaml file with class names, load them here for reporting
CLASS_NAMES = ['white', 'black']  # Replace with actual class names, e.g., from data.yaml
# If unknown, set CLASS_NAMES = [f'class{i}' for i in range(max_classes + 1)]

def load_labels(label_path):
    """
    Load YOLO format labels: [[class, x, y, w, h], ...]
    """
    if not os.path.exists(label_path):
        return np.array([])
    with open(label_path, 'r') as f:
        labels = [list(map(float, line.strip().split())) for line in f]
    return np.array(labels)  # shape: (n, 5) where 5 = [cls, x, y, w, h]

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in xywh format (normalized).
    box: [x_center, y_center, width, height]
    """
    # Convert to xyxy
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xmin1, ymin1 = x1 - w1 / 2, y1 - h1 / 2
    xmax1, ymax1 = x1 + w1 / 2, y1 + h1 / 2
    xmin2, ymin2 = x2 - w2 / 2, y2 - h2 / 2
    xmax2, ymax2 = x2 + w2 / 2, y2 + h2 / 2

    # Intersection
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate_predictions(preds, gts, iou_thres=0.5):
    """
    For a single image and single class: Compute TP, FP, FN based on IoU.
    preds: [[x, y, w, h, conf, cls], ...] sorted by conf descending
    gts: [[cls, x, y, w, h], ...]
    Returns: tp (array of 1/0 for each pred), num_gt
    """
    if len(preds) == 0:
        return np.array([]), len(gts)

    # Sort preds by confidence descending
    preds = preds[preds[:, 4].argsort()[::-1]]

    tp = np.zeros(len(preds))
    gt_matched = np.zeros(len(gts), dtype=bool)

    for i, pred in enumerate(preds):
        best_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(gts):
            if gt_matched[j]:
                continue
            iou = compute_iou(pred[:4], gt[1:])  # gt[1:] = xywh, pred[:4]=xywh
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_thres and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = True

    num_gt = len(gts)
    return tp, num_gt

def compute_ap(recall, precision):
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    i = np.where(recall[1:] != recall[:-1])[0]
    ap = sum((recall[i + 1] - recall[i]) * precision[i + 1] for i in i)
    return ap

def main():
    # Initialize detector
    detector = YoloDetecter(
        weights=WEIGHTS,
        data=DATA,
        imgsz=IMG_SIZE,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        device=DEVICE,
        classes=CLASSES
    )

    # Get list of validation images
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not img_files:
        print(f"No images found in {IMG_DIR}")
        return

    # Determine number of classes
    max_cls = 0
    for img_file in img_files:
        label_file = os.path.join(LABEL_DIR, img_file.rsplit('.', 1)[0] + '.txt')
        if os.path.exists(label_file):
            gts = load_labels(label_file)
            if len(gts) > 0:
                max_cls = max(max_cls, int(gts[:, 0].max()))
    num_classes = max_cls + 1
    print(f"Detected {num_classes} classes.")

    global CLASS_NAMES
    if len(CLASS_NAMES) != num_classes:
        CLASS_NAMES = [f'class{i}' for i in range(num_classes)]

    # Global collectors
    class_results = [[] for _ in range(num_classes)]  # list of (conf, tp)
    total_gts = np.zeros(num_classes, dtype=int)

    # Single loop for efficiency (batch processing sequentially)
    for img_file in tqdm(img_files, desc="Evaluating"):
        img_path = os.path.join(IMG_DIR, img_file)
        label_path = os.path.join(LABEL_DIR, img_file.rsplit('.', 1)[0] + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        try:
            _, pred_boxes = detector.detect(img)
        except Exception as e:
            print(f"Inference failed for {img_path}: {e}")
            continue

        pred_boxes = np.array(pred_boxes) if pred_boxes else np.array([])

        gt_boxes = load_labels(label_path)

        for cls in range(num_classes):
            cls_pred = pred_boxes[pred_boxes[:, 5] == cls] if len(pred_boxes) > 0 else np.array([])
            cls_gt = gt_boxes[gt_boxes[:, 0] == cls] if len(gt_boxes) > 0 else np.array([])

            total_gts[cls] += len(cls_gt)

            if len(cls_pred) == 0:
                continue

            tp, _ = evaluate_predictions(cls_pred, cls_gt, EVAL_IOU_THRES)
            confs = cls_pred[:, 4]

            for c, t in zip(confs, tp):
                class_results[cls].append((c, t))

    # Compute AP and mAP
    aps = []
    for cls in range(num_classes):
        results = class_results[cls]
        if total_gts[cls] == 0 or len(results) == 0:
            ap = 0.0
            aps.append(ap)
            print(f"{CLASS_NAMES[cls]}: AP = {ap:.4f} (GT={total_gts[cls]}, Preds={len(results)})")
            continue

        results.sort(key=lambda x: x[0], reverse=True)

        cum_tp = 0
        cum_fp = 0
        precisions = []
        recalls = []

        for conf, is_tp in results:
            if is_tp:
                cum_tp += 1
            else:
                cum_fp += 1
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / total_gts[cls]
            precisions.append(precision)
            recalls.append(recall)

        ap = compute_ap(np.array(recalls), np.array(precisions))
        aps.append(ap)

        print(f"{CLASS_NAMES[cls]}: AP@{EVAL_IOU_THRES:.2f} = {ap:.4f} (GT={total_gts[cls]}, TP={cum_tp}, FP={cum_fp})")

    map_val = np.mean(aps)
    print(f"\nmAP@{EVAL_IOU_THRES:.2f} = {map_val:.4f}")

    total_tp = sum(sum(tp for conf, tp in res) for res in class_results)
    total_gt_all = sum(total_gts)
    total_fp = sum(len(res) for res in class_results) - total_tp
    if total_gt_all > 0:
        overall_recall = total_tp / total_gt_all
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        overall_accuracy = total_tp / total_gt_all * 100  # Recognition accuracy as percentage (equivalent to recall * 100)
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall F1: {overall_f1:.4f}")
        print(f"Overall Recognition Accuracy: {overall_accuracy:.2f}%")

if __name__ == '__main__':
    main()