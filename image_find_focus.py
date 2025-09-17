# -*- coding: utf-8 -*-
import cv2
import numpy as np

class FocusFinder(object):
    def __init__(self, scale=1, allowed_moving_girth=300):
        self.scale = scale
        self.allowed_moving_girth = allowed_moving_girth
        self.allowed_moving_length = 10
        self.pre_corner_point = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.pre_max_length = 0
        self.is_first = True

    def find_focus(self, img, min_threshold=30, max_threshold=250):
        source_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"images/gray.jpg", gray)
        # 计算亮度（0-1范围）
        mean_brightness = np.mean(gray) / 255.0
        darkness = int(mean_brightness * 150)
        # if mean_brightness:
            # print("当前亮度过低，建议进行补光！")
        # elif mean_brightness >0.8:
            # print("当前亮度过高，建议等摄像头稳定或关闭补光！")
        # print("当前亮度为：",mean_brightness,"(0-1范围)")

        # img = cv2.GaussianBlur(gray, (3, 3), 0, 0)
        # cv2.imwrite(f"images/img_17.jpg", img)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([0, 0, darkness])
        upper_yellow = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # img = cv2.GaussianBlur(img, (5, 5), 0, 0)
        cv2.imwrite(f"images/hsv.jpg", img)
        # cv2.imshow('hsv Image', img)
        canny = cv2.Canny(img, min_threshold, max_threshold)
        cv2.imwrite(f"images/canny_19.jpg", canny)
        # cv2.imshow("canny", canny)
        k = np.ones((3, 3), np.uint8)
        canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, k)
        cv2.imwrite(f"images/canny_24.jpg", canny)

        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        if len(contours) == 0:
            return 0, False
        max_length = abs(cv2.arcLength(contours[0], True))
        # print(max_length)
        if max_length < 1000:
            return 0, False
        temp_caver = np.ones(canny.shape, np.uint8) * 255
        contours1 = cv2.approxPolyDP(contours[0], 50, True)
        cv2.drawContours(temp_caver, contours1, -1, (0, 255, 0), 1)
        cv2.imwrite(f"images/drawContours-37.jpg", temp_caver)
        # 备选拐点
        corners = cv2.goodFeaturesToTrack(temp_caver, 4, 0.6, 200)
        if corners is None:
            return 0, False
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(temp_caver, corners, (11, 11), (-1, -1), criteria)
        corners = np.int0(corners)
        point_list = []
        for i in corners:
            x, y = i.ravel()
            point_list.append((x, y))

        corner_point = self.find_corner(point_list)  # 找到4个顶点
        sort_corner_list = self.sort_corner(corner_point)

        if self.is_first:
            self.pre_corner_point = sort_corner_list
            self.pre_max_length = max_length
            self.is_first = False

        if abs(self.pre_max_length - max_length) > self.allowed_moving_girth:
            # LOG.debug("物体入侵")
            self.pre_max_length = max_length
            return 0, False

        if np.max(abs(np.array(sort_corner_list) - np.array(self.pre_corner_point))) > self.allowed_moving_length:
            # LOG.warning(f"角点错误:{sort_corner_list}, {self.pre_corner_point}")
            self.pre_corner_point = sort_corner_list
            return 0, False

        self.pre_corner_point = sort_corner_list
        self.pre_max_length = max_length

        # hight, width = self.calSize(sort_corner_list, self.scale)
        # aim_size = np.float32([[0, 0], [width, 0], [width, hight], [0, hight]])
        hight, width = 500,500
        aim_size = np.float32([[0, 0], [width, 0], [width, hight], [0, hight]])
        raw_size = []

        for x, y in sort_corner_list:
            raw_size.append([x, y])

        raw_size = np.float32(raw_size)
        translate_map = cv2.getPerspectiveTransform(raw_size, aim_size)
        translate_img = cv2.warpPerspective(source_img, translate_map, (int(width), int(hight)))
        translate_img = cv2.flip(translate_img, 1)  # 对角镜像
        return translate_img, True

    def calSize(self, sort_corner_list, scale):
        h1 = (sort_corner_list[2][1] - sort_corner_list[1][1])
        h2 = (sort_corner_list[3][1] - sort_corner_list[0][1])
        hight = max(h1, h2) * scale

        w1 = (sort_corner_list[0][0] - sort_corner_list[1][0])
        w2 = (sort_corner_list[3][0] - sort_corner_list[2][0])
        width = max(w1, w2) * scale

        return hight, width

    def sort_corner(self, corner_point):
        for i in range(len(corner_point)):
            for j in range(i + 1, len(corner_point)):
                if corner_point[i][1] > corner_point[j][1]:
                    tmp = corner_point[j]
                    corner_point[j] = corner_point[i]
                    corner_point[i] = tmp
        top = corner_point[:2]
        bot = corner_point[2:]

        if top[0][0] > top[1][0]:
            tmp = top[1]
            top[1] = top[0]
            top[0] = tmp

        if bot[0][0] > bot[1][0]:
            tmp = bot[1]
            bot[1] = bot[0]
            bot[0] = tmp

        tl = top[1]
        tr = top[0]
        bl = bot[0]
        br = bot[1]
        corners = [tl, tr, bl, br]
        return corners

    def area(self, a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])

    def find_corner(self, point_list):
        corner_num = len(point_list)
        ans = 0.0
        ans_point_index_list = [0, 0, 0, 0]
        m1_point = 0
        m2_point = 0
        for i in range(corner_num):
            for j in range(corner_num):
                if (i == j):
                    continue
                m1 = 0.0
                m2 = 0.0

                for k in range(corner_num):
                    if (k == i or k == j):
                        continue
                    a = point_list[i][1] - point_list[j][1]
                    b = point_list[j][0] - point_list[i][0]
                    c = point_list[i][0] * point_list[j][1] - point_list[j][0] * point_list[i][1]
                    temp = a * point_list[k][0] + b * point_list[k][1] + c

                    if (temp > 0):
                        tmp_area = abs(self.area(point_list[i], point_list[j], point_list[k]) / 2)
                        if tmp_area > m1:
                            m1 = tmp_area
                            m1_point = k

                    elif (temp < 0):
                        tmp_area = abs(self.area(point_list[i], point_list[j], point_list[k]) / 2)
                        if tmp_area > m2:
                            m2 = tmp_area
                            m2_point = k

                if (m1 == 0.0 or m2 == 0.0):
                    continue
                if (m1 + m2 > ans):
                    ans_point_index_list[0] = i
                    ans_point_index_list[1] = j
                    ans_point_index_list[2] = m1_point
                    ans_point_index_list[3] = m2_point
                    ans = m1 + m2
        ans_point_list = []
        for i in ans_point_index_list:
            ans_point_list.append(point_list[i])
        return ans_point_list
