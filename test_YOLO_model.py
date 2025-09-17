import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import Global_variables
from yolov5.detect_self import YoloDetecter
from image_find_focus import FocusFinder

# 五子棋盘参数
WIDTH_GOBANG = Global_variables.WIDTH_GOBANG  # 五子棋盘总宽度
LENGTH_GOBANG = Global_variables.LENGTH_GOBANG  # 五子棋盘总长度
WIDTH_ERR_GOBANG = Global_variables.WIDTH_ERR_GOBANG  # 五子棋盘内外边框间距(宽度方向)
LENGTH_ERR_GOBANG = Global_variables.LENGTH_ERR_GOBANG  # 五子棋盘内外边框间距(长度方向)
ROW_GOBANG = Global_variables.ROW_GOBANG  # 五子棋盘行数(宽度方向)
COLUMN_GOBANG = Global_variables.COLUMN_GOBANG  # 五子棋盘列数(长度方向)

def yolo_to_pixel(yolo_list, rows_b, cols_b):
    """将YOLO检测坐标转换为像素坐标"""
    data = []
    for x, y, w, h, conf, c in yolo_list:
        pixel_y = y * cols_b
        pixel_x = x * rows_b
        data.append([pixel_x, pixel_y, conf, c])
    return data

def coordinate_mapping(pixel_list, physical_rows, physical_cols, pixel_rows, pixel_cols):
    """将像素坐标映射到棋盘物理坐标"""
    data = []
    for x, y, conf, c in pixel_list:
        x = x * physical_rows / pixel_rows
        y = y * physical_cols / pixel_cols
        data.append([x, y, conf, c])
    return data

def coordinate_to_pos(coordinate_list, go_stones):
    """将物理坐标转换为棋盘行列坐标"""
    pos_set = set()
    ai_pos_set = set()
    pos_set_conf = []
    ai_pos_set_conf = []
    if go_stones == "white":
        player_class = 1
    else:
        player_class = 0

    for coordinate_x, coordinate_y, conf, c in coordinate_list:
        pos_x = round(
            abs(coordinate_x - WIDTH_ERR_GOBANG) / (WIDTH_GOBANG - 2 * WIDTH_ERR_GOBANG) * (ROW_GOBANG - 1))
        pos_y = round(
            abs(coordinate_y - LENGTH_ERR_GOBANG) / (LENGTH_GOBANG - 2 * LENGTH_ERR_GOBANG) * (COLUMN_GOBANG - 1))

        if c == player_class:  # 计算玩家的棋子
            pos_set.add((pos_x, pos_y))
            pos_set_conf.append((pos_x, pos_y, float(conf)))
        else:
            ai_pos_set.add((pos_x, pos_y))
            ai_pos_set_conf.append((pos_x, pos_y, float(conf)))
    return pos_set, ai_pos_set, pos_set_conf, ai_pos_set_conf

def detect_and_visualize(image_path, model_path, device='cpu', go_stones='black'):
    """检测test.jpg中的棋子位置，绘制bounding box和置信度"""
    # 初始化YOLO模型和FocusFinder
    self_yolo = YoloDetecter(weights=model_path, device=device)
    focus_finder = FocusFinder()

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像文件")
        return

    # 图像校正和目标区域提取
    # focus_image, has_res = focus_finder.find_focus(image)
    # if not has_res:
    #     print("无法找到棋盘焦点")
    #     return

    # YOLO检测
    res_img, yolo_list = self_yolo.detect(image)
    img_shape = res_img.shape

    # 坐标转换
    pixel_list = yolo_to_pixel(yolo_list, res_img.shape[0], res_img.shape[1])
    coordinate_list = coordinate_mapping(pixel_list, WIDTH_GOBANG, LENGTH_GOBANG, img_shape[0], img_shape[1])
    pos_set, ai_pos_set, pos_set_conf, ai_pos_set_conf = coordinate_to_pos(coordinate_list, go_stones)

    # 绘制bounding box和置信度
    for x, y, conf, c in pixel_list:
        # 计算bounding box的坐标
        box_x = int(x - yolo_list[0][2] * img_shape[1] / 2)
        box_y = int(y - yolo_list[0][3] * img_shape[0] / 2)
        box_w = int(yolo_list[0][2] * img_shape[1])
        box_h = int(yolo_list[0][3] * img_shape[0])

        # 绘制bounding box
        color = (0, 255, 0) if c == 0 else (255, 0, 0)  # 黑色棋子:绿色框，白色棋子:蓝色框
        cv2.rectangle(res_img, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)

        # 绘制置信度
        label = f"Conf: {conf:.2f}"
        cv2.putText(res_img, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 保存并显示结果
    output_path = get_root() +"/images/detected_chess_pieces.jpg"
    cv2.imwrite(output_path, res_img)
    print(f"检测结果已保存至: {output_path}")

    # 显示图像
    cv2.imshow("YOLO Detection Result", res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 获取模型路径
    def get_root():
        file = Path(__file__).resolve()
        parent_dir = file.parent
        root = str(parent_dir).replace("\\", "/")
        return root

    model_path = get_root() + "/yolov5/runs/train/exp5/weights/best.pt"
    image_path =get_root() + "/images/20250730_110851.jpg"
    detect_and_visualize(image_path, model_path, device='cpu', go_stones='black')