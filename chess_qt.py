import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread,QTimer
from PyQt5.QtGui import QImage, QPainter, QColor 
import time

# ================= 新增Qt界面部分 =================
class BoardSignal(QObject):
    update_signal = pyqtSignal(tuple)  # 用于发送坐标更新信号

class GomokuBoard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("五子棋棋盘")
        self.setFixedSize(600, 600)
        self.board_size = 13
        self.cell_size = 40
        self.board_image = QImage(self.size(), QImage.Format_ARGB32)
        self.board_image.fill(Qt.white)
        self.init_board()
        self.board_state = [[0]*9 for _ in range(9)]

    def init_board(self):
        painter = QPainter(self.board_image)
        painter.setPen(QColor(0, 0, 0))
        
        # 绘制网格线
        for i in range(self.board_size):
            start_pos = i * self.cell_size + 20
            painter.drawLine(20, start_pos, 20 + 8*self.cell_size, start_pos)
            painter.drawLine(start_pos, 20, start_pos, 20 + 8*self.cell_size)
        
        # 绘制星位
        star_points = [(4,4), (6,6), (4,2), (2,4), (2,2)]
        painter.setBrush(QColor(0, 0, 0))
        for x, y in star_points:
            pos = x * self.cell_size + 20
            painter.drawEllipse(pos-2, pos-2, 4, 4)
        
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.board_image)
        for i in range(9):
            for j in range(9):
                if self.board_state[i][j]:
                    self.draw_stone(painter, i, j, Qt.black if self.board_state[i][j] == 1 else Qt.white)

    def draw_stone(self, painter, x, y, color):
        center_x = x * self.cell_size + 20
        center_y = y * self.cell_size + 20
        painter.setPen(QColor(0, 0, 0))
        painter.setBrush(color)
        painter.drawEllipse(center_x-18, center_y-18, 36, 36)

    def update_board(self, pos):
        x, y, player = pos
        self.board_state[x][y] = 1 if player == "black" else 2
        self.update()

# ================= 主程序修改部分 =================
class MainProgram(QObject):
    def __init__(self):
        super().__init__()
        self.board_signal = BoardSignal()
        
        # 创建棋盘对象（在主线程）
        self.gomoku_board = GomokuBoard()
        self.gomoku_board.show()
        
        # 连接信号到主线程的槽函数
        self.board_signal.update_signal.connect(self.gomoku_board.update_board)

if __name__ == '__main__':
    # 必须在主线程创建 QApplication
    app = QApplication(sys.argv)
    
    # 初始化主程序
    main_program = MainProgram()
    
    # 测试信号发送（确保在界面显示后执行）
    def test_update():
        for i in range(9):
            main_program.board_signal.update_signal.emit((i, i, "black"))
    
    # 使用定时器确保界面初始化完成
    QTimer.singleShot(100, test_update)
    
    sys.exit(app.exec_())