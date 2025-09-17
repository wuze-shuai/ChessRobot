import cv2
import numpy as np
import time
import sys
import os

# 将dir2的路径添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Global_variables
from ql_main import ql_main
# 五子棋棋盘参数
ROW_GOBANG = 13  # 五子棋盘行数(宽度方向)
COLUMN_GOBANG = 13  # 五子棋盘列数(长度方向)

# 初始化棋盘状态
Global_variables.black = np.zeros((ROW_GOBANG, COLUMN_GOBANG), dtype=int)
Global_variables.white = np.zeros((ROW_GOBANG, COLUMN_GOBANG), dtype=int)
Global_variables.flag = np.zeros((ROW_GOBANG, COLUMN_GOBANG), dtype=int)

# 自动下棋函数
def auto_play():
    for i in range(ROW_GOBANG):
        for j in range(COLUMN_GOBANG):
            # 模拟玩家下棋
            Global_variables.black[i][j] = 1
            Global_variables.flag[i][j] = 1
            print(f"玩家下棋：({i}, {j})")
            #使用的是Glabal_variables里的坐标文件。
            ql_main(i+6, j,True)  # 调用QL学习算法下棋
            time.sleep(15)  # 暂停15秒，方便观察

if __name__ == '__main__':
    auto_play()