# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import sys
sys.path.append('../../')
import gdal, cv2
from libs.label_name_dict.label_dict import NAME_LABEL_MAP
from libs.configs import cfgs
from libs.box_utils import iou_rotate
from libs.box_utils import coordinate_convert
from help_utils import tools

ANGLE_TRESHOLD = 15 #15
SHOW_ALARM_LIST = True
def _write_voc_results_file(all_boxes, test_imgid_list, det_save_path):
  for cls, cls_ind in NAME_LABEL_MAP.items():
    if cls == 'back_ground':
      continue
    print('Writing {} VOC results file'.format(cls))

    with open(det_save_path, 'wt') as f:
      for im_ind, index in enumerate(test_imgid_list):
        dets = all_boxes[cls_ind][im_ind]
        if dets == []:
          continue
        # the VOCdevkit expects 1-based indices
        for k in range(dets.shape[0]):
          f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                  format(index, dets[k, -1],
                         dets[k, 0] + 1, dets[k, 1] + 1,
                         dets[k, 2] + 1, dets[k, 3] + 1, dets[k, 4] + 1))


def write_voc_results_file(all_boxes, test_imgid_list, det_save_dir):
  '''

  :param all_boxes: is a list. each item reprensent the detections of a img.
  the detections is a array. shape is [-1, 7]. [category, score, x, y, w, h, theta]
  Note that: if none detections in this img. that the detetions is : []

  :param test_imgid_list:
  :param det_save_path:
  :return:
  '''
  for cls, cls_id in NAME_LABEL_MAP.items():
    if cls == 'back_ground':
      continue
    print("Writing {} VOC resutls file".format(cls))

    tools.mkdir(det_save_dir)
    det_save_path = os.path.join(det_save_dir, "det_"+cls+".txt")
    with open(det_save_path, 'wt') as f:
      for index, img_name in enumerate(test_imgid_list):
        this_img_detections = all_boxes[index]

        this_cls_detections = this_img_detections[this_img_detections[:, 0] == cls_id]
        if this_cls_detections.shape[0] == 0:
          continue # this cls has none detections in this img
        for a_det in this_cls_detections:
          f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                  format(img_name, a_det[1],
                         a_det[2], a_det[3],
                         a_det[4], a_det[5], a_det[6]))  # that is [img_name, score, x, y, w, h, theta]


def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  # pass_flag = False
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    #obj_struct['pose'] = obj.find('pose').text
    #obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = 0 #int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    rbox = [int(bbox.find('x0').text), int(bbox.find('y0').text), int(bbox.find('x1').text),
            int(bbox.find('y1').text), int(bbox.find('x2').text), int(bbox.find('y2').text),
            int(bbox.find('x3').text), int(bbox.find('y3').text)]
        
    rbox = np.array([rbox], np.float32)
    rbox = coordinate_convert.back_forward_convert(rbox, with_label=False)
    obj_struct['bbox'] = rbox
    objects.append(obj_struct)

  return objects

def parse_rec_det_txt(detpath, filename, cls_name):
    """ Parse a PASCAL infer txt file """
    detfile = os.path.join(detpath, filename+".txt")
    with open(detfile, 'r') as f:
        lines = f.readlines()
    
    boxes = []
    for line in lines:
        words = line.strip().split(',')
        # print('words', words)
        outside_points_num = 0
        if words[8] == cls_name:
          box = np.int0(np.array(words[0:8], dtype=np.float32)) 
          box = list(coordinate_convert.back_forward_convert([box], False)[0])
          box.insert(0, float(words[9]))
          box.insert(0, filename)
          boxes.append(box)
    # print(boxes)       
    return(boxes)
        
        
def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def voc_eval(detpath, annopath, test_imgid_list, cls_name, ovthresh=0.5,
                 use_07_metric=False, use_diff=False):
  '''

  :param detpath:
  :param annopath:
  :param test_imgid_list: it 's a list that contains the img_name of test_imgs
  :param cls_name:
  :param ovthresh:
  :param use_07_metric:
  :param use_diff:
  :return:
  '''
  # 1. parse xml to get gtboxes

  # read list of images
  imagenames = test_imgid_list

  recs = {}
  for i, imagename in enumerate(imagenames):
    recs[imagename] = parse_rec(os.path.join(annopath, imagename+'.xml'))
    # if i % 100 == 0:
    #   print('Reading annotation for {:d}/{:d}'.format(
    #     i + 1, len(imagenames)))

  # 2. get gtboxes for this class.
  class_recs = {}
  num_pos = 0
  # if cls_name == 'person':
  #   print ("aaa")
  det_boxes = []
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == cls_name]
    bbox = np.array([x['bbox'] for x in R])
    if use_diff:
      difficult = np.array([False for x in R]).astype(np.bool)
    else:
      difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    num_pos = num_pos + sum(~difficult)  # ignored the diffcult boxes
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det} # det means that gtboxes has already been detected
 
  # 3. read the detection file
  detfile = os.path.join(detpath, "det_"+cls_name+".txt")
  
  with open(detfile, 'r') as f:
    lines = f.readlines()

  # for a line. that is [img_name, confidence, xmin, ymin, xmax, ymax]
  splitlines = [x.strip().split(' ') for x in lines if x.strip().split(' ')[0] in imagenames]  # a list that include a list
  image_ids = [x[0] for x in splitlines]  # img_id is img_name
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids) # num of detections. That, a line is a det_box.
  tp = np.zeros(nd)
  fp = np.zeros(nd)
  tp_angle_offsets = []

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]  #reorder the img_name

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]  # img_id is img_name
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        # ixmin = np.maximum(BBGT[:, 0], bb[0])
        # iymin = np.maximum(BBGT[:, 1], bb[1])
        # ixmax = np.minimum(BBGT[:, 2], bb[2])
        # iymax = np.minimum(BBGT[:, 3], bb[3])
        # iw = np.maximum(ixmax - ixmin + 1., 0.)
        # ih = np.maximum(iymax - iymin + 1., 0.)
        # inters = iw * ih
        #
        # # union
        # uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
        #        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
        #        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        #
        # overlaps = inters / uni
        overlaps = []
        angle_offsets = []
        for i in range(len(BBGT)):
          # print('bb', bb)
          # print('BBGT', BBGT[i])
          overlap = iou_rotate.iou_rotate_calculate1(np.array([bb]),
                                                      BBGT[i],
                                                      use_gpu=False)[0]
          overlaps.append(overlap)
          angle_offset = abs(bb[-1] - BBGT[i][0][-1])
          angle_offsets.append(angle_offset)
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        
      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            # print('angle offset', angle_offsets[jmax])
            if angle_offsets[jmax] < ANGLE_TRESHOLD:
              tp[d] = 1.
              R['det'][jmax] = 1
              tp_angle_offsets.append(angle_offsets[jmax])
          else:
            fp[d] = 1.
            # print(ovmax, angle_offsets[jmax])
      else:
        fp[d] = 1.
        # print(ovmax, angle_offsets[jmax])

  # 4. get recall, precison and AP
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(num_pos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)
  gt_cls_num = num_pos

  tp_angle_offsets = np.array(tp_angle_offsets)
  angle_offsets_mse =  np.mean(tp_angle_offsets**2)
  angle_offsets_rmse = np.sqrt(angle_offsets_mse)
  print('angle_offsets_mse:', angle_offsets_mse)
  print('angle_offsets_rmse:', angle_offsets_rmse)

  return rec, prec, ap, gt_cls_num, tp, fp


def do_python_eval(test_imgid_list, test_annotation_path, iou_thresh=0.5):
  #import matplotlib.colors as colors
  #import matplotlib.pyplot as plt

  AP_list = []
  recall_all = 0
  precision_all = 0
  cls_to_avg = 0
  gt_cls_num_all = 0
  tp_all = 0
  fp_all = 0
  for cls, index in NAME_LABEL_MAP.items():
    if cls == 'back_ground':
      continue
    recall, precision, AP, gt_cls_num, tp, fp = voc_eval(detpath=cfgs.INFERENCE_SAVE_PATH,
                                                        test_imgid_list=test_imgid_list,
                                                        cls_name=cls,
                                                        annopath=test_annotation_path,
                                                        ovthresh=iou_thresh
                                                        )
    # print('cls:', cls, 'gt num:', gt_cls_num)
    if np.isnan(AP):
      continue
    if AP == 0:#recall.size == 0 or precision.size == 0:
      if gt_cls_num == 0:
        continue
      else:
        recall, precision, tp, fp = [0], [0], [0], [0]
    AP_list += [AP]
    recall_all += recall[-1]
    precision_all += precision[-1]
    cls_to_avg += 1
    gt_cls_num_all += gt_cls_num
    tp_all += tp[-1]
    fp_all += fp[-1]
    print("cls : {}|| num : {}|| Recall: {} || Precison: {}|| AP: {}".format(cls, gt_cls_num,recall[-1], precision[-1], AP))

    '''
    c = colors.cnames.keys()
    c_dark = list(filter(lambda x: x.startswith('dark'), c))
    c = ['red', 'orange']
    plt.axis([0, 1.2, 0, 1])
    plt.plot(recall, precision, color=c_dark[index], label=cls)
    '''
  '''  
  plt.legend(loc='upper right')
  plt.xlabel('R')
  plt.ylabel('P')
  plt.savefig('./PR_R.png')
  '''
  print("avg recall is {}".format(recall_all/cls_to_avg))
  print("avg precision is {}".format(precision_all/cls_to_avg))
  print("avg false alarm is {}".format(fp_all/(tp_all+fp_all)))
  print("mAP is : {}".format(np.mean(AP_list)))

def voc_evaluate_detections(all_boxes, test_imgid_list, test_annotation_path):
  '''

  :param all_boxes: is a list. each item reprensent the detections of a img.

  The detections is a array. shape is [-1, 6]. [category, score, xmin, ymin, xmax, ymax]
  Note that: if none detections in this img. that the detetions is : []
  :return:
  '''

  write_voc_results_file(all_boxes, test_imgid_list=test_imgid_list,
                         det_save_dir=cfgs.EVALUATE_R_DIR)
  do_python_eval(test_imgid_list, test_annotation_path)





if __name__ == '__main__':
    
    _img_list = os.listdir(cfgs.INFERENCE_ANNOTATION_PATH)
    
    img_list = []
    for img in _img_list:
        if img.endswith('.xml'):
          img = img.split('.xml')[0]
          img_list.append(img)
    # do_python_eval(img_list, './Annotations')
    
    print('*'*80)
    print(cfgs.INFERENCE_SAVE_PATH)
    # do_python_eval(img_list, cfgs.INFERENCE_ANNOTATION_PATH, iou_thresh=0.1)
    do_python_eval(img_list, cfgs.INFERENCE_ANNOTATION_PATH, iou_thresh=0.5)
    do_python_eval(img_list, cfgs.INFERENCE_ANNOTATION_PATH, iou_thresh=0.7)
