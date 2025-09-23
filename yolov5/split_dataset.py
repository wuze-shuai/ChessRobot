import os
import random
from pathlib import Path

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8):
    """
    将图像和标签文件按照指定比例分配到训练集和验证集，并生成对应的data.yaml文件。

    Args:
        image_dir (str): 包含图像文件的目录路径。
        label_dir (str): 包含标签文件的目录路径。
        output_dir (str): 保存训练集和验证集的目录路径。
        train_ratio (float, optional): 训练集的比例。默认为0.8。
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    train_images_dir = images_dir / "train"
    val_images_dir = images_dir / "val"
    train_labels_dir = labels_dir / "train"
    val_labels_dir = labels_dir / "val"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # 获取图像文件列表
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    # 随机打乱文件顺序
    random.shuffle(image_files)

    # 计算训练集和验证集的数量
    total_images = len(image_files)
    train_size = int(total_images * train_ratio)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    # 创建train.txt和val.txt文件
    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"

    # 处理训练集
    with open(train_txt, "w") as f:
        for image_file in train_files:
            label_file = Path(label_dir) / f"{image_file.stem}.txt"
            if label_file.exists():
                f.write(f"{image_file}\n")
                # 复制图像和标签到训练集目录
                os.system(f"copy {image_file} {train_images_dir / image_file.name}")
                os.system(f"copy {label_file} {train_labels_dir / label_file.name}")

    # 处理验证集
    with open(val_txt, "w") as f:
        for image_file in val_files:
            label_file = Path(label_dir) / f"{image_file.stem}.txt"
            if label_file.exists():
                f.write(f"{image_file}\n")
                # 复制图像和标签到验证集目录
                os.system(f"copy {image_file} {val_images_dir / image_file.name}")
                os.system(f"copy {label_file} {val_labels_dir / label_file.name}")

    # 生成data.yaml文件
    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write("train: ./dataset/images/train\n")
        f.write("val: ./dataset/images/val\n\n")
        f.write("nc: 2\n")  # 假设有2个类别，根据实际情况修改
        f.write("names: ['class1', 'class2']\n")  # 假设有2个类别，根据实际情况修改

    print(f"数据集已成功分割到: {output_dir}")
    print(f"训练集图像数量: {len(train_files)}")
    print(f"验证集图像数量: {len(val_files)}")

# 示例用法
if __name__ == "__main__":
    image_dir = r"E:\13project\yolov5-master\data\images\black_white"  # 图像目录
    label_dir = r"E:\13project\yolov5-master\data\labels"  # 标签目录
    output_dir = r"E:\13project\yolov5-master\data\dataset"  # 输出目录
    split_dataset(image_dir, label_dir, output_dir)