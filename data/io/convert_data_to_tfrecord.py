# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
from libs.label_name_dict.label_dict import *
from help_utils.tools import *

tf.app.flags.DEFINE_string('VOC_dir', '../dataset', 'Voc dir')
tf.app.flags.DEFINE_string('xml_dir', 'Annotations', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'JPEGImages', 'image dir')
# tf.app.flags.DEFINE_string('xml_dir', 'xml', 'xml dir')
# tf.app.flags.DEFINE_string('image_dir', 'img', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', '/share/rbh/tfrecord/research_r2/head_aug_8x/','save name')#../tfrecord/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.jpg', 'format of image')
tf.app.flags.DEFINE_string('dataset', cfgs.DATASET_NAME, 'dataset')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    name = child_item.text
                    if name == 'background' or name == 'back_ground':
                        # name = 'back_ground'
                        print('back ground, pass:', name)
                        break
                    if name not in NAME_LABEL_MAP.keys():
                        print('undefined label, pass:', name)
                        break
                    if name == 'others':
                        print('negative sample')
                    label = NAME_LABEL_MAP[name]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        if node.tag not in ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']:
                            continue
                        tmp_box.append(int(node.text))
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)
    if len(box_list) == 0:
        print('this img contain empty box, pass')
        return None, None, None

    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_height, img_width, gtbox_label


def convert_pascal_to_tfrecord():
    xml_path = FLAGS.VOC_dir + FLAGS.xml_dir
    image_path = FLAGS.VOC_dir + FLAGS.image_dir
    save_path = FLAGS.save_dir + FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord'
    mkdir(FLAGS.save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)
    print(xml_path, image_path)
    for count, xml in enumerate(glob.glob(xml_path + '/*.xml')):
        
        # to avoid path error in different development platform
        xml = xml.replace('\\', '/')

        # img_name = xml.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_name = xml.split('/')[-1][0:-4] + FLAGS.img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue
        
        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)
        if gtbox_label is None :
            print('pass error img')
            continue
        #if gtbox_label.shape[0] <= 0:
        #    continue
        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)

        feature = tf.train.Features(feature={
            # do not need encode() in linux
            'img_name': _bytes_feature(img_name.encode()),
            # 'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)
        if img.shape[2] != 3:
            print('channel error!')
        if gtbox_label.shape[0] == 0:
            print('empty error!')
        
            print('*'*40)
            print('step', count)
            print('name', img_name)
            print('gt', gtbox_label.shape)
            print('img', img.shape)
        writer.write(example.SerializeToString())
        
        # print('img_path:', img_path)
        view_bar('Conversion progress', count + 1, len(glob.glob(xml_path + '/*.xml')))

    print('\nConversion is complete!')


if __name__ == '__main__':
    # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'
    # read_xml_gtbox_and_label(xml_path)

    convert_pascal_to_tfrecord()
