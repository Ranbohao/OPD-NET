'''
@Author: your name
@Date: 2020-01-06 15:43:12
@LastEditTime : 2020-01-06 16:39:27
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ranbohao/R2/research_r2/libs/networks/mda/binary_map.py
'''
import numpy as np
import cv2
import tensorflow as tf
from libs.configs import cfgs
def binary_map_generate(img_batch, gtboxes):

    def binary_map_gen(img, boxes):
        temp = np.zeros([img.shape[0], img.shape[1]], np.uint8)
        boxes = boxes.astype(np.int32)
        if cfgs.DETECT_HEAD:
            for box in boxes:
                points = np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], np.int32)
                head_points = get_head_part_points(
                    ship_points=points, head_part_ratio=cfgs.PAN_HEAD_RATIO)
                # print('all', points)
                points = points.reshape((-1, 1, 2))
                cv2.fillPoly(temp, [points], 1)

                head_points = head_points.reshape((-1, 1, 2))
                cv2.fillPoly(temp, [head_points], 2)
        else:
            for box in boxes:
                points = np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.fillPoly(temp, [points], 1)

        return temp

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(binary_map_gen,
                                       inp=[img_tensor, gtboxes],
                                       Tout=[tf.uint8])
    img_shape = tf.shape(img_tensor)
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, [1, img_shape[0], img_shape[1], 1])

    return img_tensor_with_boxes


def get_head_part_points(ship_points, head_part_ratio=0.5):
    def _get_divide_point(start_point, end_point, ratio=0.5):
        offset = (end_point - start_point)*ratio
        new_end_point = start_point + offset
        return np.array(start_point), np.array(new_end_point)

    left_top = ship_points[0]
    right_top = ship_points[1]
    left_bottom = ship_points[3]
    right_bottom = ship_points[2]

    left_top, left_bottom = _get_divide_point(
        left_top, left_bottom, ratio=head_part_ratio)
    right_top, right_bottom = _get_divide_point(
        right_top, right_bottom, ratio=head_part_ratio)
    head_part_points = np.array(
        [left_top, right_top, right_bottom, left_bottom], np.int32)
    return head_part_points
    
