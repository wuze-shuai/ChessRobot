import numpy as np

def generate_board_txt(points):
    """
    根据四个已知点生成整个棋盘的坐标文件
    :param points: 四个已知点的坐标，格式为{(x, y): (X, Y, Z)}
    :return: None，生成board.txt文件
    """
    # 提取四个已知点的坐标
    p00 = np.array(points[(0, 0)])
    p013 = np.array(points[(0, 12)])
    p1313 = np.array(points[(12, 12)])
    p130 = np.array(points[(12, 0)])

    # 计算两个方向的向量
    v1 = (p130 - p00) / 12
    v2 = (p013 - p00) / 12

    # 创建一个字典来存储棋盘的坐标
    board = {}

    # 遍历棋盘的每个点
    for i in range(13):  
        for j in range(13): 
            # 计算当前点的坐标
            x = p00[0] + i * v1[0] + j * v2[0]
            y = p00[1] + i * v1[1] + j * v2[1]
            z = p00[2] + i * v1[2] + j * v2[2]
            board[f"{i},{j}"] = f"{x:.1f}/{y:.1f}/{z:.1f}"

    # 将字典转换为字符串
    board_str = "{\n"
    for key, value in board.items():
        board_str += f'    "{key}": "{value}",\n'
    board_str = board_str.rstrip(",\n") + "\n}"

    # 写入到board.txt文件
    with open("board.txt", "w") as f:
        f.write(board_str)

# 示例调用
points = {
    (0, 0): (170.0, -160.0, 63.0),
    (0, 12): (177.0, 127.0, 63.0),
    (12, 12): (435.0, 113.0, 60.0),
    (12, 0): (428.0, -162.0, 60.0)
}

generate_board_txt(points)