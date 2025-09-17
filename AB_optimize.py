#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化后的Alpha-Beta剪枝算法，支持三种难度级别
日期：2025/07/30
"""

import copy
import Global_variables
import Calcu_every_step_score

col_num = Global_variables.COL_NUM

# 全局变量
black_used_pos = []
white_used_pos = []
search_range = []
best_pos = []

def alpha_beta_process(mod,go_stones=None):
    """
    接口函数，接收难度模式（'简单'、'中等'、'困难'），返回最佳落子位置
    """
    global search_range, best_pos
    search_range = shrink_range()
    if mod == '简单':
        best_pos, white_max_score, black_max_score = machine_thinking(mod, depth=1)
    elif mod == '中等':
        best_pos , white_max_score, black_max_score= machine_thinking_twice(mod, depth=2)
    elif mod == '困难':
        best_pos , white_max_score, black_max_score= machine_thinking_twice(mod, depth=3)
    else:
        raise ValueError("无效的难度模式")
    if go_stones == None:
        return best_pos
    else:
        return best_pos, white_max_score, black_max_score

def shrink_range():
    """
    优化搜索范围，仅考虑靠近已有棋子的位置
    """
    cover_range = [[0 for _ in range(col_num)] for _ in range(col_num)]
    for i in range(col_num):
        for j in range(col_num):
            if Global_variables.flag[i][j] == 1:
                # 扩展到周围8个格子
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < col_num and 0 <= nj < col_num:
                            cover_range[ni][nj] = 1
    # 排除已占用位置
    for i in range(col_num):
        for j in range(col_num):
            if Global_variables.flag[i][j] == 1:
                cover_range[i][j] = 0
    return cover_range

def machine_thinking(mod, depth):
    """
    简单难度：单步贪心选择
    """
    black_max_score = -float('inf')
    white_max_score = -float('inf')
    b_best_pos = None
    w_best_pos = None

    for i in range(col_num):
        for j in range(col_num):
            if Global_variables.flag[i][j] == 0 and search_range[i][j] == 1:
                Global_variables.flag[i][j] = 1
                search_range[i][j] = 0

                # 计算白棋得分
                Global_variables.white[i][j] = 1
                white_score = (Calcu_every_step_score.cal_score_wise('white', i, j)
                              if mod == '困难' else
                              Calcu_every_step_score.cal_score('white', i, j))
                Global_variables.white[i][j] = 0

                # 计算黑棋得分
                Global_variables.black[i][j] = 1
                black_score = (Calcu_every_step_score.cal_score_wise('black', i, j)
                              if mod == '困难' else
                              Calcu_every_step_score.cal_score('black', i, j))
                Global_variables.black[i][j] = 0

                Global_variables.flag[i][j] = 0
                search_range[i][j] = 1

                if black_score > black_max_score:
                    black_max_score = black_score
                    b_best_pos = (i, j)
                if white_score > white_max_score:
                    white_max_score = white_score
                    w_best_pos = (i, j)

    # 简单模式优先进攻
    if white_max_score >= 100000 or white_max_score > black_max_score:
        return w_best_pos, white_max_score, black_max_score
    return b_best_pos, white_max_score, black_max_score

def machine_thinking_twice(mod, depth):
    """
    中等和困难难度：两步或三步搜索，结合攻防策略
    """
    black_max_score = -float('inf')
    white_max_score = -float('inf')
    b_best_pos = None
    w_best_pos = None

    for i in range(col_num):
        for j in range(col_num):
            if Global_variables.flag[i][j] == 0 and search_range[i][j] == 1:
                Global_variables.flag[i][j] = 1
                search_range[i][j] = 0

                # 计算白棋得分
                Global_variables.white[i][j] = 1
                white_score = (Calcu_every_step_score.cal_score_wise('white', i, j)
                              if mod == '困难' else
                              Calcu_every_step_score.cal_score('white', i, j))
                Global_variables.white[i][j] = 0

                # 计算黑棋得分
                Global_variables.black[i][j] = 1
                black_score = (Calcu_every_step_score.cal_score_wise('black', i, j)
                              if mod == '困难' else
                              Calcu_every_step_score.cal_score('black', i, j))
                Global_variables.black[i][j] = 0

                Global_variables.flag[i][j] = 0
                search_range[i][j] = 1

                if black_score > black_max_score:
                    black_max_score = black_score
                    b_best_pos = (i, j)
                if white_score > white_max_score:
                    white_max_score = white_score
                    w_best_pos = (i, j)

    # 确定优先走法
    if white_max_score >= 100000:
        return w_best_pos, white_max_score, black_max_score
    if black_max_score >= 8000:
        return b_best_pos, white_max_score, black_max_score
    first_best = w_best_pos if white_max_score > black_max_score else b_best_pos
    second_best = b_best_pos if white_max_score > black_max_score else w_best_pos

    # 进行多步搜索
    first_sums, first_best = deeper_search(first_best, second_best, mod, depth - 1)
    second_sums, second_best = deeper_search(second_best, first_best, mod, depth - 1)

    return first_best if first_sums >= second_sums else second_best  , white_max_score, black_max_score

def deeper_search(first_best, second_best, mod, depth):
    """
    递归搜索后续步骤
    """
    if depth <= 0:
        return 0, first_best

    (x, y) = first_best
    color = 'white' if first_best in [(i, j) for i, j in white_used_pos] else 'black'
    Global_variables.flag[x][y] = 1
    Global_variables.__setitem__(color, x, y, 1)
    score = (Calcu_every_step_score.cal_score_wise(color, x, y)
             if mod == '困难' else
             Calcu_every_step_score.cal_score(color, x, y))

    global search_range
    search_range = shrink_range()
    next_pos,white_max_score,black_max_score = machine_thinking(mod, depth=1)
    next_color = 'black' if color == 'white' else 'white'
    (nx, ny) = next_pos
    Global_variables.flag[nx][ny] = 1
    Global_variables.__setitem__(next_color, nx, ny, 1)
    next_score = (Calcu_every_step_score.cal_score_wise(next_color, nx, ny)
                  if mod == '困难' else
                  Calcu_every_step_score.cal_score(next_color, nx, ny))

    # 恢复状态
    Global_variables.__setitem__(color, x, y, 0)
    Global_variables.__setitem__(next_color, nx, ny, 0)
    Global_variables.flag[x][y] = Global_variables.flag[nx][ny] = 0

    return score + next_score, first_best

# 假设Global_variables支持__setitem__操作
setattr(Global_variables, '__setitem__', lambda color, i, j, val: setattr(Global_variables, color, [[val if (r, c) == (i, j) else Global_variables.__getattribute__(color)[r][c] for c in range(col_num)] for r in range(col_num)]))