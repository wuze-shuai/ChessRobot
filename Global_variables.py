#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/12/5 22:58        
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction：  加分规则匹配正确性于2017.12.6 9:42测试完毕
                斜线加分测试与2017.12.6 17:12调试完毕√ ━━━━━☆*°☆*°
"""
# 最后还要加一个边界处理
import re

# WIDTH_GOBANG = 160  # 五子棋盘总宽度
# LENGTH_GOBANG = 160  # 五子棋盘总长度
# WIDTH_ERR_GOBANG = 0  # 五子棋盘内外边框间距(宽度方向)
# LENGTH_ERR_GOBANG = 0  # 五子棋盘内外边框间距(长度方向)
# ROW_GOBANG = 9  # 五子棋盘行数(宽度方向)
# COLUMN_GOBANG = 9  # 五子棋盘列数(长度方向)
# HIGH_GOBANG = 8  # 五子棋棋盘+棋子高度

# COL_NUM = 9

# WIDTH_GOBANG = 267  # 五子棋盘总宽度
# LENGTH_GOBANG = 267  # 五子棋盘总长度
# WIDTH_ERR_GOBANG = 40  # 五子棋盘内外边框间距(宽度方向)
# LENGTH_ERR_GOBANG = 40  # 五子棋盘内外边框间距(长度方向)
# ROW_GOBANG = 9  # 五子棋盘行数(宽度方向)
# COLUMN_GOBANG = 9  # 五子棋盘列数(长度方向)
# HIGH_GOBANG = 8  # 五子棋棋盘+棋子高度

COL_NUM = 13

WIDTH_GOBANG = 303  # 五子棋盘总宽度
LENGTH_GOBANG = 322  # 五子棋盘总长度
WIDTH_ERR_GOBANG = 17  # 五子棋盘内外边框间距(宽度方向)
LENGTH_ERR_GOBANG = 20  # 五子棋盘内外边框间距(长度方向)
ROW_GOBANG = 13  # 五子棋盘行数(宽度方向)
COLUMN_GOBANG = 13  # 五子棋盘列数(长度方向)
HIGH_GOBANG = 8  # 五子棋棋盘+棋子高度


# 五子棋物理参数
# WIDTH_GOBANG = 290  # 五子棋盘总宽度
# LENGTH_GOBANG = 290  # 五子棋盘总长度
# WIDTH_ERR_GOBANG = 20  # 五子棋盘内外边框间距(宽度方向)
# LENGTH_ERR_GOBANG = 20  # 五子棋盘内外边框间距(长度方向)
# ROW_GOBANG = 13  # 五子棋盘行数(宽度方向)
# COLUMN_GOBANG = 13  # 五子棋盘列数(长度方向)
# HIGH_GOBANG = 8  # 五子棋棋盘+棋子高度

# COL_NUM = 13


black = [[0 for a in range(COL_NUM)] for b in range(COL_NUM)]
white = [[0 for a in range(COL_NUM)] for b in range(COL_NUM)]
flag = [[0 for a in range(COL_NUM)] for b in range(COL_NUM)]
# 100000
pattern_5 = [re.compile(r'11111')]
# 10000 加了最后一个 bug修复*2 把011112移到下面
pattern_alive_4 = [re.compile(r'011110')]
# 8000 改了最后一个去了2 重大bug修复 去除重复模式
pattern_to_4 = [re.compile(r'11011'), re.compile(r'011112'), re.compile(r'10111'), re.compile(r'201111')]
# 5000 双活三原本是01110 但此处应该再加边缘2个0 长度尽量长限制足够大就不会误判
pattern_double_alive_3 = [re.compile(r'0011100'), re.compile(r'2011100')]
# 1000
pattern_alive_sleep_3 = [re.compile(r'0011102')]
# 200
pattern_alive_3 = [re.compile(r'010110')]
# 100 加了边缘两个0的限制,新增‘001102’,'001012
pattern_double_alive_2 = [re.compile(r'001100'), re.compile(r'001102'), re.compile(r'001012')]
# 50
pattern_sleep_3 = [re.compile(r'001112'), re.compile(r'010112'), re.compile(r'011012')
, re.compile(r'10011'), re.compile(r'10101'), re.compile(r'2011102')]
# 10 加了两个，无对方棋在边缘的活二
pattern_alive_sleep_2 = [re.compile(r'0010100'), re.compile(r'00100100')]
# 5
pattern_alive_2 = [re.compile(r'201010'), re.compile(r'2010010'),  re.compile(r'20100102'),  re.compile(r'2010102')]
# 3 加了两个,要保证不陷入死4，即起码还有5个空位
pattern_sleep_2 = [re.compile(r'000112'), re.compile(r'001012'), re.compile(r'010012')
, re.compile(r'10001'), re.compile(r'2010102'), re.compile(r'2011002')]
# -5 边缘一个子也会设定为 -5 ,这个可以先看一下效果
pattern_dead_4 = [re.compile(r'2\d{3}12'), re.compile(r'2\d{2}1\d{2}2')]
# -5
pattern_dead_3 = [re.compile(r'2\d{2}12')]
# -5
pattern_dead_2 = [re.compile(r'2\d12')]

all_patterns = [pattern_5, pattern_alive_4, pattern_to_4, pattern_double_alive_3, pattern_alive_sleep_3, pattern_alive_3
, pattern_double_alive_2, pattern_sleep_3, pattern_alive_sleep_2, pattern_alive_2, pattern_sleep_2, pattern_dead_4,
pattern_dead_3, pattern_dead_2]

all_scores = [100000, 10000, 8000, 5000, 1000, 200, 100, 50, 10, 5, 3, -5, -5, -5]

board_scores = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
                [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0],
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

search_range = []

filename = "./board/board.txt"

SOUND_PATH = "sound/"         # 音频文件地址

UART_PORT = "COM20"
# ESP32_IP =  "192.168.40.184"

remind_time = 10        # 提示“该你落子了”时间间隔
start_time = None