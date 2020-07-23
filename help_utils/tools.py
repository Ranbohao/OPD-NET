'''
@Author: your name
@Date: 2020-01-31 10:58:15
@LastEditTime : 2020-01-31 11:00:39
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ranbohao/R2/research_r2/help_utils/tools.py
'''
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import sys
import os
import tensorflow as tf
import tfplot as tfp
# import imageio
# import cv2


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def add_heatmap(feature_maps, name=None, is_training=True):
    '''

    :param feature_maps:[B, H, W, C]
    :return:
    '''

    def figure_attention(activation):
        fig, ax = tfp.subplots()
        im = ax.imshow(activation, cmap='jet')
        fig.colorbar(im)
        return fig

    heatmap = tf.reduce_sum(feature_maps, axis=-1)
    heatmap = tf.squeeze(heatmap, axis=0)
    if is_training:
        tfp.summary.plot(name, figure_attention, [heatmap])
    else:
        return heatmap
# def video2gif(video_file):
#     cap = cv2.VideoCapture(video_file)
#     imgs = []
#     ret,frame = cap.read()
#     frame = frame[:, :, ::-1]
#     while ret:
#         imgs.append(frame)
#         ret, frame = cap.read()
#         try:
#             frame = frame[:, :, ::-1]
#         except TypeError:
#             break
#     imageio.mimsave(video_file.replace(video_file.split('.')[-1],'gif'), imgs[10:],duration=1/20.0)

# if __name__ == '__main__':
#     video_file = './demo1.avi'
#     video2gif(video_file)