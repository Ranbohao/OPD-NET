'''
@Author: your name
@Date: 2019-11-01 14:35:42
@LastEditTime : 2020-02-03 15:54:06
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ranbohao/R2/research_r2/libs/configs/cfgs.py
'''
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf


# ------------------------------------------------
VERSION = 'opd_net_cpcrm_pan_multi_fb_v1'
NET_NAME = 'resnet_v1_101'
ADD_BOX_IN_TENSORBOARD = True
# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0"
SHOW_TRAIN_INFO_INTE = 1
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 10000 #5000

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'

INFERENCE_IMAGE_PATH = '/share/rbh/dataset/hrsc/head/Test/bak/AllImages'
INFERENCE_SAVE_PATH = '/share/rbh/dataset/hrsc/head/Test/bak/eval/new_hrsc_head_dfpn_ori_lr001_2'
INFERENCE_ANNOTATION_PATH = '/share/rbh/dataset/hrsc/head/Test/bak/Annotations'

if NET_NAME.startswith('resnet'):
    weights_name = NET_NAME
elif NET_NAME.startswith('MobilenetV2'):
    weights_name = 'mobilenet/mobilenet_v2_1.0_224'
else:
    raise NotImplementedError

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# EVALUATE_H_DIR = ROOT_PATH + '/output' + '/evaluate_h_result_pickle/' + VERSION
# EVALUATE_R_DIR = ROOT_PATH + '/output' + '/evaluate_r_result_pickle/' + VERSION
# TEST_IMAGES_PATH = '/home/share/dataset/tianzhi/labeled_data/research/head/JPEGImages_test/'
# TEST_ANNOTATION_PATH = '/home/share/dataset/tianzhi/labeled_data/research/head/Annotations_test/'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False # True
IS_FILTER_OUTSIDE_BOXES = False# True
ROTATE_NMS_USE_GPU = True
FIXED_BLOCKS = 2  # allow 0~3

RPN_LOCATION_LOSS_WEIGHT = 1 / 7
RPN_CLASSIFICATION_LOSS_WEIGHT = 2.0

PAN_LOSS_WEIGHT = 1.0 #1
FAST_RCNN_LOCATION_LOSS_WEIGHT = 4.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 2.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0

MUTILPY_BIAS_GRADIENT = None  # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 30.0#None   # 10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.0003 # 0.0001
DECAY_STEP = [30000, 40000]
MAX_ITERATION = 50000

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'HRSC' #'ship' #'DOTA'  # 'ship', 'spacenet', 'pascal', 'coco'
DATASETS = ['HRSC', 'tianzhi_new', 'tianzhi', 'marine', 'ship', 'DOTA', 'ship', 'spacenet', 'pascal', 'coco', 'head']
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 800
CLASS_NUM = 1

# --------------------------------------------- Network_config
BATCH_SIZE = 1
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.0001


# ---------------------------------------------Anchor config
BASE_ANCHOR_SIZE_LIST = [256]  # can be modified
ANCHOR_STRIDE = [16]  # can not be modified in most situations
ANCHOR_SCALES = [0.125, 0.25, 0.5, 1., 2.0]
ANCHOR_RATIOS = [1, 1 / 3., 3., 5., 1 / 5., 7., 1 / 7., 9., 1 / 9]

ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0, 10.0]#15.0]
ANCHOR_SCALE_FACTORS = None


# --------------------------------------------RPN config
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.5# 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.2#0.3
IS_RPN_NEGATIVE_NOT_SMALLER_ZERO = True 
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.7# 0.5
RPN_NMS_IOU_THRESHOLD = 0.7#0.9#0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 5000 # 10000  # 5000
RPN_MAXIMUM_PROPOSAL_TEST = 1000 #2000#300  # 300

# -------------------------------------------Fast-RCNN config
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0
SHOW_SCORE_THRSHOLD = 0.5

FAST_RCNN_NMS_IOU_THRESHOLD = 0.1
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 500
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 256
FAST_RCNN_POSITIVE_RATE = 0.5

ADD_GTBOXES_TO_TRAIN = False

DETECT_HEAD = True

USE_CPCRM = True
RBB_LEN = 6 if USE_CPCRM else 5

USE_PAN = True
PAN_MORE_CONV = True
PAN_HEAD_RATIO = 0.25
PAN_MULTIPLY = True  # True

USE_FEATURE_BALANCE = True

USE_LIGHT_HEAD = False