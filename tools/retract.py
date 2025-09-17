import numpy as np
import json
import sys
import os
# 将dir2的路径添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Global_variables
from ql_main import ser
# 从文件中加载字典
def load_dict_from_file(filename):
    with open(filename, 'r') as f:
        loaded_data = json.load(f)
    # 将字符串格式的键转换回元组
    return {(int(k.split(',')[0]), int(k.split(',')[1])): v for k, v in loaded_data.items()}

# 将相对位置转换为棋盘上的绝对位置
def relative_to_absolute_position(relative_pos, points):
    """
    将相对位置转换为棋盘上的绝对位置
    :param relative_pos: 相对位置，格式为 (x_ratio, y_ratio)，范围在 [0, 1] 之间
    :param points: 四个已知点的坐标，格式为{(x, y): (X, Y, Z)}
    :return: 绝对位置 (x, y, z)
    """
    # 提取四个已知点的坐标
    p00 = np.array(points[(0, 0)].split('/'), dtype=float)
    p012 = np.array(points[(0, 12)].split('/'), dtype=float)
    p1212 = np.array(points[(12, 12)].split('/'), dtype=float)
    p120 = np.array(points[(12, 0)].split('/'), dtype=float)

    # 计算两个方向的向量
    v1 = (p120 - p00) / 12  # x方向的单位向量
    v2 = (p012 - p00) / 12  # y方向的单位向量

    # 提取相对位置的比例
    x_ratio, y_ratio = relative_pos

    # 计算绝对位置
    absolute_pos = p00 + x_ratio * 12 * v1 + y_ratio * 12 * v2

    return absolute_pos



def retract(x,y):
    restored_board = load_dict_from_file(Global_variables.filename)
    relative_pos = (x, y)
    absolute_pos = relative_to_absolute_position(relative_pos, restored_board)
    x = f"{absolute_pos[0]:.1f}"
    y = f"{absolute_pos[1]:.1f}"
    z = f"{absolute_pos[2]:.1f}"
    flag = True
    result = x+'/'+y+'/'+z+'/'+str(flag)
    ser.write(result.encode())
    print("Sent:", result)

    
# 示例调用
retract(0.492000013589859, 0.5139999985694885)