import json
import Global_variables
import serial
import socket
# ESP32_PORT = 8081
# ESP32_IP = Global_variables.ESP32_IP
# # # # 创建 socket1
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# client_socket.connect((ESP32_IP, ESP32_PORT))

# ESP32 的 IP 地址和端口
UART_PORT = Global_variables.UART_PORT
try:
    ser = serial.Serial(
        port=UART_PORT,
        baudrate=9600,
        timeout=1,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )
except serial.serialutil.SerialException as e:
    print(f"警告：无法正确连接机械臂端口 {UART_PORT}，机械臂功能将禁用。错误: {e}")
    ser = None  # 连接失败，设置 ser 为 None 作为备用
except Exception as e:
    print(f"警告：发生未知错误，机械臂功能将禁用。错误: {e}")
    ser = None  # 捕获其他潜在异常

# 从文本文件读取并恢复为字典
def load_dict_from_file(filename):
    with open(filename, 'r') as f:
        loaded_data = json.load(f)
    # 将字符串格式的键转换回元组
    return {(int(k.split(',')[0]), int(k.split(',')[1])): v for k, v in loaded_data.items()}

# 从文件中读取字典
def ql_main(board_x,board_y,retract=False):
    restored_board = load_dict_from_file(Global_variables.filename)
    print(restored_board[(board_x, board_y)])
    result=restored_board[(board_x, board_y)]+'/'+str(retract)
    # client_socket.send(result.encode())  # 无线发送数据
    # ser.write(result.encode())    #有线发送数据
    print("Sent:", result)