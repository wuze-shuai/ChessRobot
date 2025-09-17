
import math

import numpy as np
import time
import json
import socket
import time
import Global_variables
import serial

    # 机械臂底座高度
base_height = 150
    # 机械臂大摆臂长度
big_arm_length = 150
    # 机械臂小摆臂长度
small_arm_length = 150
    # 机械臂末端长度
end_length = 50

# ESP32_PORT = 8081
# ESP32_IP = Global_variables.ESP32_IP
# # # # 创建 socket1
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# client_socket.connect((ESP32_IP, ESP32_PORT))

# ESP32 的 IP 地址和端口
UART_PORT = Global_variables.UART_PORT
ser = serial.Serial(
            port=UART_PORT,
            baudrate=9600,
            timeout=1,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )

# 从文本文件读取并恢复为字典
def load_dict_from_file(filename):
    with open(filename, 'r') as f:
        loaded_data = json.load(f)
    # 将字符串格式的键转换回元组
    return {(int(k.split(',')[0]), int(k.split(',')[1])): v for k, v in loaded_data.items()}


# 从文件中读取字典

def ql_main(board_x,board_y,retract=False):
    if retract == False:
        restored_board = load_dict_from_file(Global_variables.filename)
        # if int(restored_board[(board_x, board_y)]) == 0:
        #     print("ok")
        #     end_stop_x, end_stop_y = get_xy_position_by_board(board_x, board_y)
        #     end_stop_z = 145
        #     target_degree_1, target_degree_2, target_degree_3 = get_degree_by_xyz_position(end_stop_x, end_stop_y,
        #                                                                                    end_stop_z)
        #     result = str(target_degree_1) + '/' + str(target_degree_2) + '/' + str(target_degree_3)
        # else:
        #     print(restored_board[(board_x, board_y)])
        #     result=restored_board[(board_x, board_y)]
        print(restored_board[(board_x, board_y)])
        result=restored_board[(board_x, board_y)]+'/'+str(retract)
    else:
        restored_board = load_dict_from_file(Global_variables.filename)
        print(restored_board[(board_x, board_y)])
        result=restored_board[(board_x, board_y)]+'/'+str(retract)
    # client_socket.send(result.encode())  # 发送数据
    ser.write(result.encode())
    print("Sent:", result)

def get_xy_position_by_board(x,y):
    pos_x=20*x+120
    pos_y=20*y-80
    return pos_x, pos_y

#通过x,y,z坐标得出3个轴的角度
def get_degree_by_xyz_position(x, y, z):
    """根据xyz坐标，计算出各个轴的角度"""
    # 计算第1轴的角度
    degree_1 = math.degrees(math.atan(abs(y) / x))
    # 得到去除末端长度的x、y、z值
    x = x - math.cos(math.radians(degree_1)) * end_length
    if y < 0:
        y = -(abs(y) - math.sin(math.radians(degree_1)) * end_length)
        degree_1 = -degree_1  # 正负值的设定
    else:
        y = abs(y) - math.sin(math.radians(degree_1)) * end_length

    # 判断末端的Z值与底座的Z值大小，从而调用不同的函数进行处理
    temp_length = math.sqrt(x ** 2 + y ** 2)  # 计算出投影在xy平面的 "三角形"底边的长
    if z == base_height:  # 末端与底座水平
        # 大摆臂的角度（未加90°）
        #print(temp_length,big_arm_length)
        degree_2 = math.degrees(math.acos((temp_length ** 2) / (2 * temp_length * big_arm_length)))
        # 小摆臂的角度
        degree_3 = 180 - 2 * degree_2
        # 大摆臂加上90度
        degree_2 += 90
    elif z > base_height:  # 末端高于底座
        # 大摆臂角度
        degree_2_1 = math.degrees(math.atan((z - base_height) / temp_length))
        temp_length_2 = math.sqrt(temp_length ** 2 + (z - base_height) ** 2)
        degree_2_2 = math.degrees(math.acos((temp_length_2 ** 2) / (2 * big_arm_length * temp_length_2)))
        degree_2 = degree_2_1 + degree_2_2 + 90
        # 小摆臂角度
        degree_3 = 180 - 2 * degree_2_2
    else:  # 末端低于底座
        # 大摆臂角度
        degree_2_1 = math.degrees(math.atan(temp_length / (base_height - z)))
        temp_length_2 = math.sqrt(temp_length ** 2 + (base_height - z) ** 2)
        degree_2_2 = math.degrees(math.acos((temp_length_2 ** 2) / (2 * big_arm_length * temp_length_2)))
        degree_2 = degree_2_1 + degree_2_2  # 注意此时不要加90
        # 小摆臂角度
        degree_3 = 180 - 2 * degree_2_2

    return round(degree_1,2), round(degree_2,2), round(degree_3,2)
if __name__ == '__main__':
    while True:
        a0 = input("请输棋盘的x'：")
        if a0.lower() == 'q':  # 如果用户输入 'q'，退出循环
            print("退出程序。")
            break
        a1 = input("请输棋盘的y'：")
        ql_main(int(a0),int(a1))