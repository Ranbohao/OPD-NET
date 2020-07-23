# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
sys.path.append("../")

if __name__ == '__main__':
    from libs.configs import cfgs
    if cfgs.DETECT_HEAD:
        from libs.val_libs import voc_eval_r_head as voc_eval_r
    else:
        from libs.val_libs import voc_eval_r

    _anno_list = os.listdir(cfgs.INFERENCE_ANNOTATION_PATH)
    
    anno_list = []
    for anno in _anno_list:
        if anno.endswith('.xml'):
          anno = anno.split('.xml')[0]
          anno_list.append(anno)
    
    print('*'*80)
    print(cfgs.INFERENCE_SAVE_PATH)
    voc_eval_r.do_python_eval(anno_list, cfgs.INFERENCE_ANNOTATION_PATH, iou_thresh=0.5)
    voc_eval_r.do_python_eval(anno_list, cfgs.INFERENCE_ANNOTATION_PATH, iou_thresh=0.7)


















