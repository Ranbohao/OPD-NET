# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import math


def decode_boxes(encode_boxes, reference_boxes, scale_factors=None):
    '''

    :param encoded_boxes:[N, 4]
    :param reference_boxes: [N, 4] .
    :param scale_factors: use for scale.

    in the first stage, reference_boxes  are anchors
    in the second stage, reference boxes are proposals(decode) produced by first stage
    :return:decode boxes [N, 4]
    '''

    t_xcenter, t_ycenter, t_w, t_h = tf.unstack(encode_boxes, axis=1)
    if scale_factors:
        t_xcenter /= scale_factors[0]
        t_ycenter /= scale_factors[1]
        t_w /= scale_factors[2]
        t_h /= scale_factors[3]

    reference_xmin, reference_ymin, reference_xmax, reference_ymax = tf.unstack(
        reference_boxes, axis=1)
    # reference boxes are anchors in the first stage

    reference_xcenter = (reference_xmin + reference_xmax) / 2.
    reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin

    predict_xcenter = t_xcenter * reference_w + reference_xcenter
    predict_ycenter = t_ycenter * reference_h + reference_ycenter
    predict_w = tf.exp(t_w) * reference_w
    predict_h = tf.exp(t_h) * reference_h

    predict_xmin = predict_xcenter - predict_w / 2.
    predict_xmax = predict_xcenter + predict_w / 2.
    predict_ymin = predict_ycenter - predict_h / 2.
    predict_ymax = predict_ycenter + predict_h / 2.

    return tf.transpose(tf.stack([predict_xmin, predict_ymin,
                                  predict_xmax, predict_ymax]))


def encode_boxes(unencode_boxes, reference_boxes, scale_factors=None):
    '''

    :param unencode_boxes: [-1, 4]
    :param reference_boxes: [-1, 4]
    :return: encode_boxes [-1, 4]
    '''

    xmin, ymin, xmax, ymax = unencode_boxes[:, 0], unencode_boxes[:,
                                                                  1], unencode_boxes[:, 2], unencode_boxes[:, 3]

    reference_xmin, reference_ymin, reference_xmax, reference_ymax = reference_boxes[:, 0], reference_boxes[:, 1], \
        reference_boxes[:, 2], reference_boxes[:, 3]

    x_center = (xmin + xmax) / 2.
    y_center = (ymin + ymax) / 2.
    w = xmax - xmin + 1e-8
    h = ymax - ymin + 1e-8

    reference_xcenter = (reference_xmin + reference_xmax) / 2.
    reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin + 1e-8
    reference_h = reference_ymax - reference_ymin + 1e-8

    # w + 1e-8 to avoid NaN in division and log below

    t_xcenter = (x_center - reference_xcenter) / reference_w
    t_ycenter = (y_center - reference_ycenter) / reference_h
    t_w = np.log(w/reference_w)
    t_h = np.log(h/reference_h)

    if scale_factors:
        t_xcenter *= scale_factors[0]
        t_ycenter *= scale_factors[1]
        t_w *= scale_factors[2]
        t_h *= scale_factors[3]

    return np.transpose(np.stack([t_xcenter, t_ycenter, t_w, t_h], axis=0))

# using angle to present direction
def decode_boxes_rotate(encode_boxes, reference_boxes, scale_factors=None):
    '''

    :param encode_boxes:[N, 5]
    :param reference_boxes: [N, 5] .
    :param scale_factors: use for scale
    in the rpn stage, reference_boxes are anchors
    in the fast_rcnn stage, reference boxes are proposals(decode) produced by rpn stage
    :return:decode boxes [N, 5]
    '''

    t_xcenter, t_ycenter, t_w, t_h, t_theta = tf.unstack(encode_boxes, axis=1)
    if scale_factors:
        t_xcenter /= scale_factors[0]
        t_ycenter /= scale_factors[1]
        t_w /= scale_factors[2]
        t_h /= scale_factors[3]
        t_theta /= scale_factors[4]

    reference_xmin, reference_ymin, reference_xmax, reference_ymax = tf.unstack(reference_boxes, axis=1)
    reference_x_center = (reference_xmin + reference_xmax) / 2.
    reference_y_center = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin
    reference_theta = tf.ones(tf.shape(reference_xmin)) * -90
    predict_x_center = t_xcenter * reference_w + reference_x_center
    predict_y_center = t_ycenter * reference_h + reference_y_center
    predict_w = tf.exp(t_w) * reference_w
    predict_h = tf.exp(t_h) * reference_h
    predict_theta = t_theta * 180 / math.pi + reference_theta
    decode_boxes = tf.transpose(tf.stack([predict_x_center, predict_y_center,
                                          predict_w, predict_h, predict_theta]))
    return decode_boxes

# using angle to present direction
def encode_boxes_rotate(unencode_boxes, reference_boxes, scale_factors=None):
    '''
    :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 5]
    :param reference_boxes: [H*W*num_anchors_per_location, 5]
    :return: encode_boxes [-1, 5]
    '''
    x_center, y_center, w, h, theta = \
        unencode_boxes[:, 0], unencode_boxes[:, 1], unencode_boxes[:, 2], unencode_boxes[:, 3], unencode_boxes[:, 4]
    reference_xmin, reference_ymin, reference_xmax, reference_ymax = \
        reference_boxes[:, 0], reference_boxes[:, 1], reference_boxes[:, 2], reference_boxes[:, 3]
    reference_x_center = (reference_xmin + reference_xmax) / 2.
    reference_y_center = (reference_ymin + reference_ymax) / 2.
    # here maybe have logical error, reference_w and reference_h should exchange,
    # but it doesn't seem to affect the result.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin
    reference_theta = np.ones(reference_xmin.shape) * -90
    reference_w += 1e-8
    reference_h += 1e-8
    w += 1e-8
    h += 1e-8  # to avoid NaN in division and log below
    t_xcenter = (x_center - reference_x_center) / reference_w
    t_ycenter = (y_center - reference_y_center) / reference_h
    t_w = np.log(w / reference_w)
    t_h = np.log(h / reference_h)
    t_theta = (theta - reference_theta) * math.pi / 180
    if scale_factors:
        t_xcenter *= scale_factors[0]
        t_ycenter *= scale_factors[1]
        t_w *= scale_factors[2]
        t_h *= scale_factors[3]
        t_theta *= scale_factors[4]
    return np.transpose(np.stack([t_xcenter, t_ycenter, t_w, t_h, t_theta]))



if __name__ == '__main__':
    unencode_boxes = np.array([[10., 10., 20., 10., -75.],
                               [10., 10., 20., 10., -76.],
                               [10., 10., 20., 10., -77.],
                               [10., 10., 20., 10., -78.],
                               [10., 10., 20., 10., -79.],
                               [10., 10., 20., 10., -80.],
                               [10., 10., 20., 10., -81.],
                               [10., 10., 20., 10., -82.],
                               [10., 10., 20., 10., -83.],
                               [10., 10., 20., 10., -84.],
                               [10., 10., 20., 10., -85.],
                               [10., 10., 20., 10., -86.],
                               [10., 10., 20., 10., -87.],
                               [10., 10., 20., 10., -88.],
                               [10., 10., 20., 10., -89.],
                               [10., 10., 20., 10., -90.],
                               [10., 10., 20., 10., -91.],
                               [10., 10., 20., 10., -92.],
                               [10., 10., 20., 10., -93.], ], dtype=np.float32)
    reference_boxes = np.array([[0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.],
                                [0., 10., 10., 10.], ], dtype=np.float32)
    target_boxes_r = encode_boxes_rotate(
        unencode_boxes, reference_boxes, scale_factors=None)
    output_boxes_r = decode_boxes_rotate(target_boxes_r, reference_boxes)
    # print(output_boxes_r)

    a = 0.
    b = -1.
    div = tf.atan(tf.truediv(a, b))
    with tf.Session() as sess:
        print(sess.run(div))
        print('target_boxes_r')
        # print(target_boxes_r)
        #print(sess.run([output_boxes_r[:, -3]]))
        #print(sess.run([output_boxes_r[:, -2]]))
        print(sess.run([output_boxes_r[:, -1]]))
