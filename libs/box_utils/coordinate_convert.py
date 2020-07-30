# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from libs.configs import cfgs

import cv2
import numpy as np


def forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    boxes = []
    if with_label:
        for rect in coordinate:
            theta = rect[4]
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), theta))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            theta = rect[4]
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), theta))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    return np.array(boxes, dtype=np.float32)


def back_forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)] 
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)
            # output angle range:(-180, 180)
            if cfgs.DETECT_HEAD:
                if rect1[1][0] <= rect1[1][1]:
                    rect1 = list(rect1)
                    rect1[1] = list(rect1[1])
                    temp = rect1[1][0]
                    rect1[1][0] = rect1[1][1]
                    rect1[1][1] = temp
                    rect1[2] = rect1[2] - 90
                x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

                x_head = (box[0][0] + box[1][0]) / 2
                y_head = (box[0][1] + box[1][1]) / 2
                x_c = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4
                y_c = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4
                if theta > -90:
                    if x_head - x_c > 0 and y_head - y_c < 0:
                        theta = theta
                    elif x_head - x_c < 0 and y_head - y_c > 0:
                        theta = theta + 180
                
                if theta < -90:
                    if x_head - x_c > 0 and y_head - y_c > 0:
                        theta = theta + 180
                    elif x_head - x_c < 0 and y_head - y_c < 0:
                        theta = theta
            
            else:
                if cfgs.USE_CPCRM:
                    # output angle range:(-180, 0)
                    if rect1[1][0] <= rect1[1][1]:
                        rect1 = list(rect1)
                        rect1[1] = list(rect1[1])
                        temp = rect1[1][0]
                        rect1[1][0] = rect1[1][1]
                        rect1[1][1] = temp
                        rect1[2] = rect1[2] - 90
                    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

                else:
                    # output angle range:(-90, 0)
                    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
                
            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            # output angle range:(-180, 180)
            if cfgs.DETECT_HEAD:
                if rect1[1][0] <= rect1[1][1]:
                    rect1 = list(rect1)
                    rect1[1] = list(rect1[1])
                    temp = rect1[1][0]
                    rect1[1][0] = rect1[1][1]
                    rect1[1][1] = temp
                    rect1[2] = rect1[2] - 90
                x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

                x_head = (box[0][0] + box[1][0]) / 2
                y_head = (box[0][1] + box[1][1]) / 2
                x_c = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4
                y_c = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4
                if theta > -90:
                    if x_head - x_c > 0 and y_head - y_c < 0:
                        theta = theta
                    elif x_head - x_c < 0 and y_head - y_c > 0:
                        theta = theta + 180
                
                if theta < -90:
                    if x_head - x_c > 0 and y_head - y_c > 0:
                        theta = theta + 180
                    elif x_head - x_c < 0 and y_head - y_c < 0:
                        theta = theta
            
            else:
                if cfgs.USE_CPCRM:
                    # output angle range:(-180, 0)
                    if rect1[1][0] <= rect1[1][1]:
                        rect1 = list(rect1)
                        rect1[1] = list(rect1[1])
                        temp = rect1[1][0]
                        rect1[1][0] = rect1[1][1]
                        rect1[1][1] = temp
                        rect1[2] = rect1[2] - 90
                    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

                else:
                    # output angle range:(-90, 0)
                    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
                
            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)


if __name__ == '__main__':
    coord = np.array([[0, 0, 10, 20, 0, 1],
                      [0, 0, 10, 20, 10, 1],
                      [0, 0, 10, 20, 30, 1],
                      [0, 0, 10, 20, 45, 1],
                      [0, 0, 20, 10, 0, 1],
                      [0, 0, 20, 10, 10, 1],
                      [0, 0, 20, 10, 30, 1],
                      [0, 0, 20, 10, 45, 1]])

    coord1 = np.array([[10, 5, 9, 6, -10, -5, -9, -6, 1],# -145,
                      [1, 5, 2, 6, 5, 1, 6, 2, 1],# -45 
                      [1, 2, 2, 1, 5, 6, 6, 5, 1],# -135
                      [2, 1, 2, 2, 5, 1, 5, 2, 1],# -180
                      [1, 2, 2, 1, -1, -2, -2, -1, 1]
                      ])

    coord2 = back_forward_convert(coord1)
    coord3 = forward_convert(coord2)
