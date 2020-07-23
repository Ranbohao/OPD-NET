import tensorflow.contrib.slim as slim
import tensorflow as tf

def inception_module(inputs, scope=None, reuse=None):
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'InceptionModule', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_0 = slim.conv2d(branch_0, 224, [1, 3], scope='Conv2d_0b_1x3')
                branch_0 = slim.conv2d(branch_0, 256, [3, 1], scope='Conv2d_0c_3x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 192, [5, 1], scope='Conv2d_0b_5x1')
                branch_1 = slim.conv2d(branch_1, 224, [1, 5], scope='Conv2d_0c_1x5')
                branch_1 = slim.conv2d(branch_1, 224, [7, 1], scope='Conv2d_0d_7x1')
                branch_1 = slim.conv2d(branch_1, 256, [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                branch_2 = slim.conv2d(branch_2, 128, [1, 1], scope='Conv2d_0b_1x1')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_3x3')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])