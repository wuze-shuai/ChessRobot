import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import socket
import json
from pathlib import Path
import serial
import Global_variables

board_txt = './board/board.txt'
class GobangBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.grid_size = 40
        self.board_size = 13
        self.selected_pos = (0, 0)
        self.setFixedSize(self.board_size * self.grid_size, self.board_size * self.grid_size + 10)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制棋盘
        painter.setPen(QPen(Qt.black, 2))
        painter.drawEllipse(QPoint(int(self.board_size * self.grid_size/2), 10), 8, 8)
        painter.drawText(int(self.board_size * self.grid_size/2) + 15, 15, "机械臂位置")
        for i in range(self.board_size):
            # 横线
            painter.drawLine(self.grid_size//2, self.grid_size//2 + i*self.grid_size + 20,
                            self.width()-self.grid_size//2, self.grid_size//2 + i*self.grid_size + 20)
            # 竖线
            painter.drawLine(self.grid_size//2 + i*self.grid_size, self.grid_size//2 + 20,
                            self.grid_size//2 + i*self.grid_size, self.height()-self.grid_size//2 + 10)
            
            if i == (self.board_size - 1) : 
                painter.drawText(self.grid_size//2 + (self.board_size - 1)*self.grid_size +10 , self.grid_size//2 + i*self.grid_size + 5 + 20, "X")

            else:
            
                painter.drawText(self.grid_size//2 + (self.board_size - 1)*self.grid_size +10 , self.grid_size//2 + i*self.grid_size + 5 + 20, str(i))
            
            if i == 0 : 
                painter.drawText(self.grid_size//2 + i*self.grid_size - 5, 15 + 20, "Y")

            else:
            
                painter.drawText(self.grid_size//2 + i*self.grid_size - 5, 15 +20, str((self.board_size - 1)-i))
            
        
        # 绘制选中点
        if self.selected_pos[0] != -1:
            x, y = self.selected_pos
            center_x = self.grid_size//2 + ((self.board_size - 1)-y)* self.grid_size
            center_y = self.grid_size//2 + x * self.grid_size + 20
            painter.setPen(QPen(Qt.red, 4))
            painter.drawEllipse(QPoint(center_x, center_y), 5, 5)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机械臂控制与五子棋棋盘")
        self.resize(400, 580)
        # self.ESP32_IP = "192.168.40.143"
        # self.ESP32_PORT = 8081
        # self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.client_socket.connect((self.ESP32_IP, self.ESP32_PORT))
        UART_PORT = Global_variables.UART_PORT
        self.ser = serial.Serial(
                    port=UART_PORT,
                    baudrate=9600,
                    timeout=1,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                )
        # 主控件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 上部控制面板
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # 机械臂角度输入
        arm_group = QGroupBox("机械臂角度控制")
        arm_layout = QVBoxLayout()
        
        # 调整输入框宽度
        input_style = "QLineEdit { max-width: 80px; min-width: 80px; }"
        
        x_layout = QHBoxLayout()
        self.axis1 = QLineEdit("0")
        self.axis1.setStyleSheet(input_style)
        self.btn1_plus = QPushButton('+')
        self.btn1_plus.setFixedWidth(30)
        self.btn1_minus = QPushButton('-')
        self.btn1_minus.setFixedWidth(30)
        x_layout.addWidget(self.axis1)
        x_layout.addWidget(self.btn1_plus)
        x_layout.addWidget(self.btn1_minus)

        y_layout = QHBoxLayout()
        self.axis2 = QLineEdit("0")
        self.axis2.setStyleSheet(input_style)
        self.btn2_plus = QPushButton('+')
        self.btn2_plus.setFixedWidth(30)
        self.btn2_minus = QPushButton('-')
        self.btn2_minus.setFixedWidth(30)
        y_layout.addWidget(self.axis2)
        y_layout.addWidget(self.btn2_plus)
        y_layout.addWidget(self.btn2_minus)


        z_layout = QHBoxLayout()
        self.axis3 = QLineEdit("0")
        self.axis3.setStyleSheet(input_style)
        self.btn3_plus = QPushButton('+')
        self.btn3_plus.setFixedWidth(30)
        self.btn3_minus = QPushButton('-')
        self.btn3_minus.setFixedWidth(30)
        z_layout.addWidget(self.axis3)
        z_layout.addWidget(self.btn3_plus)
        z_layout.addWidget(self.btn3_minus)

        self.btn1_plus.clicked.connect(lambda: self.adjust_value(self.axis1, 1))
        self.btn1_minus.clicked.connect(lambda: self.adjust_value(self.axis1, -1))
        self.btn2_plus.clicked.connect(lambda: self.adjust_value(self.axis2, 1))
        self.btn2_minus.clicked.connect(lambda: self.adjust_value(self.axis2, -1))
        self.btn3_plus.clicked.connect(lambda: self.adjust_value(self.axis3, 1))
        self.btn3_minus.clicked.connect(lambda: self.adjust_value(self.axis3, -1))
        
        
        # arm_layout.addWidget(self.axis1)
        # arm_layout.addWidget(self.btn_plus)
        
        arm_layout.addWidget(QLabel("第一轴角度:"))
        arm_layout.addLayout(x_layout)
        arm_layout.addWidget(QLabel("第二轴角度:"))
        # arm_layout.addWidget(self.axis2)
        arm_layout.addLayout(y_layout)
        arm_layout.addWidget(QLabel("第三轴角度:"))
        arm_layout.addLayout(z_layout)
        # arm_layout.addWidget(self.axis3)
        
        arm_group.setLayout(arm_layout)
        control_layout.addWidget(arm_group)
        
        # 棋盘坐标选择
        board_group = QGroupBox("棋盘坐标选择")
        board_layout = QVBoxLayout()
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        for i in range(13):
            self.x_combo.addItem(str(i))
            self.y_combo.addItem(str(i))
        board_layout.addWidget(QLabel("X坐标:"))
        board_layout.addWidget(self.x_combo)
        board_layout.addWidget(QLabel("Y坐标:"))
        board_layout.addWidget(self.y_combo)
        board_group.setLayout(board_layout)
        control_layout.addWidget(board_group)
        
        layout.addWidget(control_panel)

        # 添加按钮
        btn_layout = QVBoxLayout()
        self.confirm_btn = QPushButton("确定")
        self.save_btn = QPushButton("保存")
        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(self.save_btn)
        # arm_layout.addLayout(btn_layout)
        control_layout.addLayout(btn_layout)
        

        # 下部显示面板
        display_panel = QWidget()
        display_layout = QHBoxLayout(display_panel)
        
        # # 机械臂图片
        # self.arm_image = QLabel()
        # self.arm_image.setPixmap(QPixmap("robot_arm.png").scaled(400, 400, Qt.KeepAspectRatio))
        # display_layout.addWidget(self.arm_image)
        
        # 五子棋棋盘
        self.gobang_board = GobangBoard()
        display_layout.addWidget(self.gobang_board)
        
        layout.addWidget(display_panel)
        
        # 信号连接
        self.x_combo.currentIndexChanged.connect(self.update_board_selection)
        self.y_combo.currentIndexChanged.connect(self.update_board_selection)

        self.confirm_btn.clicked.connect(self.get_arm_angles)
        self.save_btn.clicked.connect(self.save_all_data)
        self.check_coordinate(board_txt,0,0)

    def adjust_value(self, line_edit, delta):
        try:
            current = float(line_edit.text())
            line_edit.setText(f"{current + delta:.2f}")
        except ValueError:
            line_edit.setText("0.00")

    def check_coordinate(self,file_path, x, y):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            key = f"{x},{y}"
            if key in data:
                # print(f"坐标({x},{y})的值是: {data[key]}")
                a, b, c = map(float, data[key].split('/'))
                self.axis1.setText(f"{a}")
                self.axis2.setText(f"{b}")
                self.axis3.setText(f"{c}")
                
                
            else:
                print(f"坐标({x},{y})不存在")
        except FileNotFoundError:
            print("文件未找到")
        except json.JSONDecodeError:
            print("文件格式错误")
    
    def update_board_selection(self):
        x = self.x_combo.currentIndex()
        y = self.y_combo.currentIndex()
        self.gobang_board.selected_pos = (x, y)
        self.check_coordinate(board_txt, x, y)
        self.gobang_board.update()

    def update_txt_file(self,data_dict, file_path):
        """将字典数据追加或覆盖到文本文件"""
        file = Path(file_path)
        
        # 读取现有内容
        existing_data = {}
        if file.exists():
            try:
                with open(file, 'r') as f:
                    existing_data = json.loads(f.read())
            except json.JSONDecodeError:
                existing_data = {}
        
        # 更新数据
        existing_data.update(data_dict)
        
        # 写入文件
        with open(file, 'w') as f:
            f.write(json.dumps(existing_data, indent=4))
    
    def get_arm_angles(self):
        try:
            # angle1 = float(self.axis1.text())
            # angle2 = float(self.axis2.text())
            # angle3 = float(self.axis3.text())
            retract = 'False'
            data = self.axis1.text() + '/' + self.axis2.text()+'/'+self.axis3.text()+'/'+ retract
            # self.client_socket.send(data.encode())
            self.ser.write(data.encode())
            QMessageBox.information(self, "角度值", f"获取角度值:\n第一轴: {self.axis1.text()}\n第二轴: {self.axis2.text()}\n第三轴: {self.axis3.text()} 已发送")
            # print(data)
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数字")
    
    def save_all_data(self):
        try:
            # angle1 = float(self.axis1.text())
            # angle2 = float(self.axis2.text())
            # angle3 = float(self.axis3.text())
            x = self.x_combo.currentIndex()
            y = self.y_combo.currentIndex()
            key = str(x) + ',' + str(y) 
            value = self.axis1.text() + '/' + self.axis2.text() +'/' + self.axis3.text()

            data = {key: value}
            #print(data)
            
            self.update_txt_file(data,board_txt)
            QMessageBox.information(self, "保存数据", 
                                  f"角度值: {self.axis1.text()}, {self.axis2.text()}, {self.axis3.text()}\n坐标: ({x}, {y}) 已保存")

        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数字")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
