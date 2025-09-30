# -*- coding: utf-8 -*-
import os
import time

import Global_variables

import Global_variables
from playsound import playsound
from random import choice, randint

col_num = Global_variables.COL_NUM

WIDTH_GOBANG = Global_variables.WIDTH_GOBANG  # 五子棋盘总宽度
LENGTH_GOBANG = Global_variables.LENGTH_GOBANG  # 五子棋盘总长度
HIGH_GOBANG = Global_variables.HIGH_GOBANG  # 五子棋棋盘+棋子高度
WIDTH_ERR_GOBANG = Global_variables.WIDTH_ERR_GOBANG  # 五子棋盘内外边框间距(宽度方向)
LENGTH_ERR_GOBANG = Global_variables.LENGTH_ERR_GOBANG  # 五子棋盘内外边框间距(长度方向)
ROW_GOBANG = Global_variables.ROW_GOBANG  # 五子棋盘行数(宽度方向)
COLUMN_GOBANG = Global_variables.COLUMN_GOBANG  # 五子棋盘列数(长度方向)


def play_sound(sound,end):
    # SOUND_PATH = ROOT + '/src/sound/'
    # SOUND_PATH = '/src/sound/'
    # for str_sound in args:
    #     path_list = []
    #     for filename in os.listdir(SOUND_PATH):
    #         if filename.startswith(str_sound):
    #             path_list.append(SOUND_PATH + filename)
    #     playsound(choice(path_list))
    if end == "wav":
        sound_path = Global_variables.SOUND_PATH + '%s.wav' % sound
        playsound(sound_path)
    elif end =="mp3":
        sound_path = Global_variables.SOUND_PATH + '%s.mp3' % sound
        playsound(sound_path)
    


def coordinate_mapping(pixel_list, physical_rows, physical_cols, pixel_rows, pixel_cols):
    # LOG.debug(f"坐标映射: {physical_rows}, {physical_cols}, {pixel_rows}, {pixel_cols}")
    data = []
    for x, y, c in pixel_list:
        x = x * physical_rows / pixel_rows
        y = y * physical_cols / pixel_cols
        data.append([x, y, c])
    return data


def coordinate_to_pos(coordinate_list, go_stones):
    pos_set = set()
    ai_pos_set = set()

    if go_stones == "white":
        player_class = 1
    else:
        player_class = 0

    for coordinate_x, coordinate_y, c in coordinate_list:
        pos_x = round(
            abs(coordinate_x - WIDTH_ERR_GOBANG) / (WIDTH_GOBANG - 2 * WIDTH_ERR_GOBANG) * (ROW_GOBANG - 1))
        pos_y = round(
            abs(coordinate_y - LENGTH_ERR_GOBANG) / (LENGTH_GOBANG - 2 * LENGTH_ERR_GOBANG) * (COLUMN_GOBANG - 1))
        
        if c == player_class:  # 计算玩家的棋子
            pos_set.add((pos_x, pos_y))
        else: 
            ai_pos_set.add((pos_x, pos_y))

    return pos_set, ai_pos_set


def pos_to_coordinate(x, y):
    x = WIDTH_ERR_GOBANG + x * (WIDTH_GOBANG - 2 * WIDTH_ERR_GOBANG) / (COLUMN_GOBANG - 1)
    y = LENGTH_ERR_GOBANG + y * (LENGTH_GOBANG - 2 * LENGTH_ERR_GOBANG) / (ROW_GOBANG - 1)
    return x, y


def Check():
    for i in range(col_num):
        for j in range(col_num-4):
            if Global_variables.black[i][j:j+5] == [1, 1, 1, 1, 1]:
                return 'black'
            elif Global_variables.white[i][j:j+5] == [1, 1, 1, 1, 1]:
                return 'white'
    for i in range(col_num):
        for j in range(col_num-4):
            if Global_variables.black[j][i] and Global_variables.black[j+1][i] and Global_variables.black[j+2][i] and Global_variables.black[j+3][i] and Global_variables.black[j+4][i]:
                return 'black'
            elif Global_variables.white[j][i] and Global_variables.white[j+1][i] and Global_variables.white[j+2][i] and Global_variables.white[j+3][i] and Global_variables.white[j+4][i]:
                return 'white'
    for i in range(col_num-4):
        for j in range(col_num-4):
            if Global_variables.black[i][j] and Global_variables.black[i+1][j+1] and Global_variables.black[i+2][j+2] and Global_variables.black[i+3][j+3] and Global_variables.black[i+4][j+4]:
                return 'black'
            elif Global_variables.white[i][j] and Global_variables.white[i+1][j+1] and Global_variables.white[i+2][j+2] and Global_variables.white[i+3][j+3] and Global_variables.white[i+4][j+4]:
                return 'white'
    for i in range(col_num-4):
        for j in range(col_num-1):
            if Global_variables.black[i][j] and Global_variables.black[i+1][j-1] and Global_variables.black[i+2][j-2] and Global_variables.black[i+3][j-3] and Global_variables.black[i+4][j-4]:
                return 'black'
            elif Global_variables.white[i][j] and Global_variables.white[i+1][j-1] and Global_variables.white[i+2][j-2] and Global_variables.white[i+3][j-3] and Global_variables.white[i+4][j-4]:
                return 'white'


def safe_detect(capture, self_yolo):
    hand_class = 'hand'
    count = 0
    while True:
        hand_flag = False
        cur_img = get_video_frame(capture)
        res_img, yolo_list = self_yolo.detect(cur_img)

        for _, _, _, _, c in yolo_list:
            if c == hand_class:
                count += 1
                if count >= 2:
                    count = 0
                    hand_flag = True
                    print("人手进入工作区域")
                    break
        if not hand_flag:
            break 

        time.sleep(0.1)


def get_video_frame(capture):
    # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
    for _ in range(10):
        ret, frame = capture.read()
    if not ret:
        print("错误，摄像头无数据")
    return frame


def compare_boards(board1, board2):
    diffs = []
    for i in range(Global_variables.COL_NUM):
        for j in range(Global_variables.COL_NUM):
            if board1[i][j] != board2[i][j]:
                diffs.append((i, j))
    return diffs


def get_current_boards(pos_set, ai_pos_set, go_stones):
    current_white = [[0 for a in range(Global_variables.COL_NUM)] for b in range(Global_variables.COL_NUM)]
    current_black = [[0 for a in range(Global_variables.COL_NUM)] for b in range(Global_variables.COL_NUM)]
    current_flag = [[0 for a in range(Global_variables.COL_NUM)] for b in range(Global_variables.COL_NUM)]

    if go_stones == "white":
        for pos_x, pos_y in ai_pos_set:
            current_white[pos_x][pos_y] = 1
            current_flag[pos_x][pos_y] = 1
        for pos_x, pos_y in pos_set:
            current_black[pos_x][pos_y] = 1
            current_flag[pos_x][pos_y] = 1
    else:
        for pos_x, pos_y in ai_pos_set:
            current_black[pos_x][pos_y] = 1
            current_flag[pos_x][pos_y] = 1
        for pos_x, pos_y in pos_set:
            current_white[pos_x][pos_y] = 1
            current_flag[pos_x][pos_y] = 1
    
    return current_black, current_white, current_flag

