# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
sys.path.append("../")
import tensorflow as tf
import time
import cv2
import numpy as np
import argparse

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network_new
from help_utils.tools import *
from libs.box_utils import draw_box_in_img
from help_utils import tools
from libs.box_utils import coordinate_convert
from libs.label_name_dict.label_dict import *
LABEl_NAME_MAP = get_label_name_map()
IS_RESIZE = False
WRITE_VOC = True
OUTPUT_H_IMG = False

def write_voc_results_file(boxes, labels, scores, img_name, det_save_dir):
    '''
    write cls.txt
    [img_name, probability, w, h, x_c, y_c, theta]
    '''
    tools.mkdir(det_save_dir)
    num, _ = boxes.shape
    for i in range(num):
        for cls, cls_id in NAME_LABEL_MAP.items():
            if cls == 'back_ground':
                continue
            if labels[i] == cls_id :
                #print("Writing {} VOC resutls file".format(cls))
                det_save_path = os.path.join(det_save_dir, "det_"+cls+".txt")
                with open(det_save_path, 'a') as f:
                    f.write('{:s} {:.6f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(img_name, scores[i],
                            boxes[i][0], boxes[i][1],
                            boxes[i][2], boxes[i][3], boxes[i][4]))  # that is [img_name, score, x, y, w, h, theta]   

def write_pixel_results(boxes, labels, scores, img_name, det_save_dir):
    '''
    write coordinate in pixel
    img_name.txt:
    [x1, y1, ..., x4, y4, category, probability]
    '''
    boxes = coordinate_convert.forward_convert(boxes, with_label=False)
    if len(labels) == 0:
        return None
        
    num, _ = boxes.shape
    with open(os.path.join(cfgs.INFERENCE_SAVE_PATH,img_name+'.txt'), 'a') as txt :
        for i in range(num):
            box = boxes[i].astype(np.int32)
            box = box.tolist()
            box.append(LABEl_NAME_MAP[labels[i]])
            box.append(scores[i])
            box.append('\n')
            txt.write(','.join([str(x) for x in box]))
            # print(box)       
                                 
def inference(det_net, data_dir):
    TIME = 0
    TIME_NUM = 0
    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     is_resize=IS_RESIZE)

    det_boxes_h, det_scores_h, det_category_h, \
    det_boxes_r, det_scores_r, det_category_r = det_net.build_whole_detection_network(input_img_batch=img_batch,
                                                                                      gtboxes_h_batch=None,
                                                                                      gtboxes_r_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        imgs = os.listdir(data_dir)
        # print(imgs)
        for i, a_img_name in enumerate(imgs):
            file_text, extension = os.path.splitext(a_img_name)
            if extension != ".jpg" and extension != ".tif" and extension != ".png":
                continue
            # f = open('./res_icdar_r/res_{}.txt'.format(a_img_name.split('.jpg')[0]), 'w')

            raw_img = cv2.imread(os.path.join(data_dir,
                                              a_img_name))
            # raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            start = time.time()
            resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
            det_boxes_r_, det_scores_r_, det_category_r_ = \
                sess.run(
                    [img_batch, det_boxes_h, det_scores_h, det_category_h,
                     det_boxes_r, det_scores_r, det_category_r],
                    feed_dict={img_plac: raw_img}
                )
            end = time.time()

            TIME += end-start
            TIME_NUM += 1

            if WRITE_VOC == True:
                boxes = np.array(det_boxes_r_, np.int64)
                scores = np.array(det_scores_r_, np.float32)
                labels = np.array(det_category_r_, np.int32)
            
                det_save_dir= cfgs.INFERENCE_SAVE_PATH
                write_voc_results_file(boxes, labels, scores,
                                   a_img_name.split('.')[0], det_save_dir)
                write_pixel_results(boxes, labels, scores,
                                   a_img_name.split('.')[0], det_save_dir)

            det_detections_h = draw_box_in_img.draw_box_cv(np.squeeze(resized_img, 0),
                                                           boxes=det_boxes_h_,
                                                           labels=det_category_h_,
                                                           scores=det_scores_h_)
            det_detections_r = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
                                                                  boxes=det_boxes_r_,
                                                                  labels=det_category_r_,
                                                                  scores=det_scores_r_)
            save_dir = os.path.join(cfgs.INFERENCE_SAVE_PATH, cfgs.VERSION)
            tools.mkdir(save_dir)
            if OUTPUT_H_IMG:
                cv2.imwrite(save_dir + '/' + a_img_name + '_h.jpg',
                            det_detections_h)
            
            cv2.imwrite(save_dir + '/' + a_img_name + '_r.jpg',
                        det_detections_r)
            view_bar('{} cost {}s'.format(a_img_name, (end - start)), i + 1, len(imgs))

        
        print('avg time:{}'.format(TIME/TIME_NUM))
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    
    parser.add_argument('--data_dir', dest='data_dir',
                        help='data path',
                        default=cfgs.INFERENCE_IMAGE_PATH, type=str)
    
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    det_net = build_whole_network_new.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                   is_training=False)
    # data_dir = cfgs.INFERENCE_IMAGE_PATH
    inference(det_net, data_dir=args.data_dir)
    # inference(det_net, data_dir)

















