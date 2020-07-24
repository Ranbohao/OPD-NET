from libs.networks.pan.pan_loss import pan_loss_cross_entropy
# from libs.networks.pan import cfgs as cfgs
from libs.configs import cfgs
from libs.networks import resnet
from libs.networks.pan.inception_module import inception_module
from help_utils.tools import add_heatmap

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

class PAN(object):
    def __init__(self, base_network_name,
                 is_training,
                 is_only_pan
                 ):
        self.is_training = is_training
        self.base_network_name = base_network_name
        self.is_only_pan = is_only_pan
    def build_base_network(self, input_img_batch):
        return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

    def build_pan_network(self, inputs, binary_map=None):
        if self.is_only_pan:
            feature_map = self.build_base_network(inputs)
        else :
            feature_map = inputs
        with tf.variable_scope('PAN_net'):
            target_shape = [tf.shape(feature_map)[1], tf.shape(feature_map)[2]]
            saliency_map, saliency_after_softmax = self.pan_net(feature_map)
            if self.is_training:
                binary_map_resize = self.get_binary_map_resize(binary_map, target_shape)
                pan_attention_loss = self.pan_loss(saliency_after_softmax, binary_map_resize)
                return self.pan_predict(saliency_after_softmax), pan_attention_loss, binary_map_resize
            else:
                return self.pan_predict(saliency_after_softmax)

    def get_binary_map_resize(self, binary_map, target_shape):
        with tf.variable_scope('resize_binary_map'):
            return tf.image.resize_nearest_neighbor(binary_map, target_shape, name="binary_map_resize")

    def pan_net(self, feature_map):
        with tf.variable_scope('pan_net'):
            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                                stride=1,
                                padding='SAME',
                                trainable=self.is_training):
                feature_map = inception_module(feature_map)               
                saliency_map = slim.conv2d(feature_map, 256, [1, 1], scope="conv_1")
                saliency_map = slim.conv2d(saliency_map, 256, [1, 1], scope="conv_2")
                saliency_map = slim.conv2d(saliency_map, 128, [1, 1], scope="conv_3")
                saliency_map = slim.conv2d(saliency_map, 128, [1, 1], scope="conv_4")
                
                saliency_map = slim.conv2d(
                        feature_map, 3, [1, 1], scope="conv_5")
                saliency_after_softmax = tf.nn.softmax(saliency_map)

                return saliency_map, saliency_after_softmax

    def pan_loss(self, saliency_after_softmax, binary_map_resize):
        with tf.variable_scope('pan_net_loss'):
            
            pan_attention_loss = pan_loss_cross_entropy(saliency_after_softmax, binary_map_resize)
            slim.losses.add_loss(pan_attention_loss)

            return pan_attention_loss

    def pan_predict(self, saliency_after_softmax):
        with tf.variable_scope('pan_predict'):
            background_feature = tf.expand_dims(
                    saliency_after_softmax[:, :, :, 0], 3)
            body_feature = tf.expand_dims(saliency_after_softmax[:,:,:,2], 3) 
            head_feature = tf.expand_dims(saliency_after_softmax[:,:,:,1], 3) 
            head_feature_1 = slim.conv2d(head_feature, 1, [1, 1], scope="conv_1")
            head_feature_2 = slim.conv2d(head_feature, 1, [3, 3], scope="conv_2")
            head_feature_3 = slim.conv2d(head_feature, 1, [1, 3], scope="conv_3")
            head_feature_4 = slim.conv2d(head_feature, 1, [3, 1], scope="conv_4")
            head_feature = head_feature_1 + head_feature_2 + head_feature_3 + head_feature_4
            predict = body_feature + head_feature

            add_heatmap(body_feature, 'img/body_feature')
            add_heatmap(head_feature, 'img/head_feature')
            add_heatmap(background_feature, 'img/background_feature')
            add_heatmap(predict, 'img/predict_mask')

            return predict
        
    def feature_map_sample(self, featrue_map):
        with tf.variable_scope('feature_map_sample'):
            feature_map_c3 = featrue_map
            feature_map_fusion = tf.reduce_mean(feature_map_c3, axis=3)
            
            feature_map_sample_1 = feature_map_c3[:,:,:,0]
            feature_map_sample_2 = feature_map_c3[:,:,:,2]
            feature_map_sample_3 = feature_map_c3[:,:,:,5]
            feature_map_sample_4 = feature_map_c3[:,:,:,7]
            feature_map_sample_5 = feature_map_c3[:,:,:,10]
            
            return feature_map_fusion, feature_map_sample_1, feature_map_sample_2, feature_map_sample_3,\
                   feature_map_sample_4, feature_map_sample_5

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            if cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()
            for var in model_variables:
                    print(var.name)
            # print(model_variables)

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith('Fast-RCNN/'+self.base_network_name+'/block4'):
                    var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    if var.name.startswith(self.base_network_name):
                        var_name_in_ckpt = name_in_ckpt_rpn(var)
                        nameInCkpt_Var_dict[var_name_in_ckpt] = var
                    else:
                        continue
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"---")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path
                        









