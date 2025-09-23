import cv2
import torch
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox

def load_classes_from_yaml(yaml_path, output_path):
    """读取 myData.yaml 文件中的类别并保存到 classes.txt"""
    yaml_path = Path(yaml_path)
    output_path = Path(output_path)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取 YAML 文件
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 提取类别名称，兼容两种格式
    names = data.get('names', [])
    if isinstance(names, dict):
        classes = [names[key] for key in sorted(names.keys())]  # 按键排序提取值
    elif isinstance(names, list):
        classes = names
    else:
        raise ValueError(f"Invalid 'names' format in {yaml_path}. Expected list or dict.")

    # 保存到 classes.txt
    with open(output_path, 'w', encoding='utf-8') as f:
        for cls in classes:
            f.write(f"{cls}\n")

    return classes

def run_yolo_detection(image_dir, save_dir, weights, data, img_size=640, conf_thres=0.25, iou_thres=0.45, device=""):
    """
    使用YOLOv5对指定目录下的所有图像进行检测，并将检测结果保存为YOLO格式的标注文件。
    """
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    img_size = check_img_size(img_size, s=stride)  # 检查图像大小

    # 获取图像列表
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    # 遍历每个图像文件
    for image_file in image_files:
        # 读取图像
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"无法读取图像: {image_file}")
            continue
        # 图像预处理
        im_pil = letterbox(image, img_size, stride=stride, auto=pt)[0]
        im_pil = im_pil.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im_pil = np.ascontiguousarray(im_pil)
        im_pil = torch.from_numpy(im_pil).to(device)
        im_pil = im_pil.half() if model.fp16 else im_pil.float()  # uint8 to fp16/32
        im_pil /= 255  # 0 - 255 to 0.0 - 1.0
        if im_pil.ndimension() == 3:
            im_pil = im_pil[None]  # expand for batch dim
        # 推理
        pred = model(im_pil, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        # 处理检测结果
        det = pred[0]
        if len(det):
            # 将边界框从img_size大小调整回原始图像大小
            det[:, :4] = scale_boxes(im_pil.shape[2:], det[:, :4], image.shape).round()
            # 保存检测结果到txt文件
            save_label_path = save_dir / f"{image_file.stem}.txt"
            with open(save_label_path, "w") as f:
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).view(-1).tolist()  # 归一化
                    cls_id = int(cls)  # 获取类别索引
                    f.write(f"{cls_id} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")
    print(f"检测结果已保存到: {save_dir}")

def check_img_size(img_size, s=32):
    """
    检查图像大小是否为stride的倍数。
    """
    new_size = (img_size + s - 1) // s * s
    if new_size != img_size:
        print(f"Warning: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}")
    return new_size

# 示例用法
if __name__ == "__main__":
    image_dir = r"E:\13project\yolov5-master\data\images\black_white"  # 图像目录
    save_dir = r"E:\13project\yolov5-master\data\labels\labels_2"  # 保存检测结果的目录
    weights = r"E:\13project\yolov5-master\runs\train\exp2_wrong_nc80\weights\best.pt"  # 模型权重文件路径
    data = r"E:\13project\yolov5-master\data\myData.yaml"  # 数据集配置文件路径
    run_yolo_detection(image_dir, save_dir, weights, data)