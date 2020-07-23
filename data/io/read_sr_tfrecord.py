# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append('../../')
from data.io import image_preprocess
from libs.configs import cfgs


def read_single_example_and_decode(filename_queue):

    # tfrecord_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    # reader = tf.TFRecordReader(options=tfrecord_options)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            
            'low_img_height': tf.FixedLenFeature([], tf.int64),
            'low_img_width': tf.FixedLenFeature([], tf.int64),
            
            'img': tf.FixedLenFeature([], tf.string),
            'low_img': tf.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
            'num_objects': tf.FixedLenFeature([], tf.int64)
        }
    )
    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    low_img_height = tf.cast(features['low_img_height'], tf.int32)
    low_img_width = tf.cast(features['low_img_width'], tf.int32)
    img = tf.decode_raw(features['img'], tf.uint8)
    low_img = tf.decode_raw(features['low_img'], tf.uint8)

    img = tf.reshape(img, shape=[img_height, img_width, 3])
    low_img = tf.reshape(low_img, shape=[low_img_height, low_img_width, 3])

    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])

    num_objects = tf.cast(features['num_objects'], tf.int32)
    return img_name, img, gtboxes_and_label, num_objects, low_img


def read_and_prepocess_single_img(filename_queue, shortside_len, is_training):

    img_name, img, gtboxes_and_label, num_objects, low_img = read_single_example_and_decode(filename_queue)

    img = tf.cast(img, tf.float32)
    img = img - tf.constant(cfgs.PIXEL_MEAN)
    low_img = tf.cast(low_img, tf.float32)
    low_img = low_img - tf.constant(cfgs.PIXEL_MEAN)
    '''
    if is_training:
        low_img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=low_img, gtboxes_and_label=gtboxes_and_label,
                                                                        target_shortside_len=shortside_len)
        # low_img, gtboxes_and_label = image_preprocess.random_flip_left_right(img_tensor=low_img,
        #                                                                      gtboxes_and_label=gtboxes_and_label)
                                                                         
        img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len)
        # img, gtboxes_and_label = image_preprocess.random_flip_left_right(img_tensor=img,
        #                                                                  gtboxes_and_label=gtboxes_and_label)

    else:
        low_img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=low_img, gtboxes_and_label=gtboxes_and_label,
                                                                        target_shortside_len=shortside_len)
        img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len)
    '''
    return img_name, img, gtboxes_and_label, num_objects, low_img


def next_batch(dataset_name, batch_size, shortside_len, is_training):
    '''
    :return:
    img_name_batch: shape(1, 1)
    img_batch: shape:(1, new_imgH, new_imgW, C)
    gtboxes_and_label_batch: shape(1, Num_Of_objects, 5] .each row is [x1, y1, x2, y2, label]
    '''
    assert batch_size == 1, "we only support batch_size is 1.We may support large batch_size in the future"

    if dataset_name not in cfgs.DATASETS:
        raise ValueError('dataSet name must be in cfgs.DATASETS')

    if is_training:
        pattern = os.path.join('../data/tfrecord', dataset_name + '_train*')
        # pattern = os.path.join(cfgs.ROOT_PATH, '/data/tfrecord', dataset_name + '_train*')
    else:
        pattern = os.path.join('../data/tfrecord', dataset_name + '_test*')
        # pattern = os.path.join(cfgs.ROOT_PATH, '/data/tfrecord', dataset_name + '_test*')

    print('tfrecord path is -->', os.path.abspath(pattern))

    filename_tensorlist = tf.train.match_filenames_once(pattern)

    filename_queue = tf.train.string_input_producer(filename_tensorlist)

    img_name, img, gtboxes_and_label, num_obs, low_img = read_and_prepocess_single_img(filename_queue, shortside_len,
                                                                                       is_training=is_training)
    img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch, low_img_batch = \
        tf.train.batch(
                       [img_name, img, gtboxes_and_label, num_obs, low_img],
                       batch_size=batch_size,
                       capacity=1,
                       num_threads=1,
                       dynamic_pad=True)
    return img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch, low_img_batch


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    os.chdir(os.path.join(cfgs.ROOT_PATH, "tools"))
    img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch, low_img_batch = \
        next_batch(dataset_name=cfgs.DATASET_NAME,  # 'pascal', 'coco'
                   batch_size=cfgs.BATCH_SIZE,
                   shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                   is_training=True)
    gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 9])

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        img_name_batch_, img_batch_, gtboxes_and_label_batch_, num_objects_batch_, low_img_batch_ \
            = sess.run([img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch, low_img_batch])

        print('debug')
        print('img_name', img_name_batch_)
        print('img', type(img_batch_))
        print('img', img_batch_.shape)
        print('low_img', low_img_batch_.shape)

        coord.request_stop()
        coord.join(threads)
