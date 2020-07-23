# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
import xml.etree.cElementTree as ET
from libs.configs import cfgs
import numpy as np
import tensorflow as tf
import glob
import cv2
import gdal
from libs.label_name_dict.label_dict import *
from help_utils.tools import *

tf.app.flags.DEFINE_string('data_dir', None, 'data_dir')

tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', cfgs.ROOT_PATH + '/data/tfrecord/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.tif', 'format of image')
tf.app.flags.DEFINE_string('dataset', cfgs.DATASET_NAME, 'dataset')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_txt_gtbox_and_label(txt_path):
    with open(txt_path,'r') as txt:
        box_list = []
        for line in txt:
            words = line.strip().split(',')
            if len(words) == 1:
                continue
            print(words)
            
            label = NAME_LABEL_MAP[words[0]]
            assert label is not None, 'label is none, error'
            
            tmp_box = []
            for i in range(1,9):
                tmp_box.append(int(words[i]))
            tmp_box.append(label)
            box_list.append(tmp_box)
            '''
            x1, y1 = (int(words[1]), int(words[2]))
            x2, y2 = (int(words[3]), int(words[4]))
            x3, y3 = (int(words[5]), int(words[6]))
            x4, y4 = (int(words[7]), int(words[8]))
            '''
        gtbox_label = np.array(box_list, dtype=np.int32)
        return gtbox_label
        
def read_tif(image_path) :
    dataset = gdal.Open(image_path)       #打开文件
    
    im_width = dataset.RasterXSize    #栅格矩阵的列数
    im_height = dataset.RasterYSize   #栅格矩阵的行数
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    
    im_data = dataset.ReadAsArray()
    data_type = im_data.dtype.__str__()
    
    if data_type != 'uint8' :
        if im_bands == 1:
            im_data = im_data.astype(np.float64)
            im_data = (im_data - im_data.min()) * 256.0 / \
                      (im_data.max() - im_data.min())
            im_data = im_data.astype(np.uint8)
            im_data = np.tile(im_data,(3,1,1))
        else :
            for i in range(im_bands):
                tmp = im_data[i, :, :].astype(np.float64)
                tmp = (tmp - tmp.min()) * 256.0 / (
                        tmp.max() - tmp.min())
                im_data[i, :, :] = tmp
                im_data = im_data.astype(np.uint8)
                im_data = im_data.swapaxes(0,2).swapaxes(0,1)[...,::-1]
    return im_width, im_height, im_data
    
def convert_pascal_to_tfrecord():
    data_path = FLAGS.data_dir
    save_path = FLAGS.save_dir + FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord'
    mkdir(FLAGS.save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)
    for count, txt in enumerate(glob.glob(data_path + '/*.txt')):
        # to avoid path error in different development platform
        txt = txt.replace('\\', '/')

        img_name = txt.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_path = data_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        gtbox_label = read_txt_gtbox_and_label(txt)

        # img = np.array(Image.open(img_path))
        # img = cv2.imread(img_path)
        img_width, img_height, img = read_tif(img_path)
        

        feature = tf.train.Features(feature={
            # do not need encode() in linux
            # 'img_name': _bytes_feature(img_name.encode()),
            'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(data_path + '/*.txt')))

    print('\nConversion is complete!')


if __name__ == '__main__':
    # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'
    # read_xml_gtbox_and_label(xml_path)

    convert_pascal_to_tfrecord()
