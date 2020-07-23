from libs.configs import cfgs
import tensorflow as tf
import tensorflow.contrib.slim as slim


def pan_loss_cross_entropy(saliency_map, binary_map):    
    binary_map = tf.squeeze(binary_map, 0) #[w / S, h / S, 1]
    binary_map = tf.reshape(binary_map, shape=[-1]) #[-1, 1]
    
    if cfgs.DETECT_HEAD:
        saliency_map = tf.reshape(saliency_map, shape=[-1, 3]) #[-1, 2]
        binary_map = tf.one_hot(binary_map, 3) #[-1, 3]
    else:
        saliency_map = tf.reshape(saliency_map, shape=[-1, 2]) #[-1, 2]
        binary_map = tf.one_hot(binary_map, 2) #[-1, 2]
    cross_entropy = - tf.reduce_sum(binary_map * tf.log(tf.clip_by_value(saliency_map, 1e-10, 1.0)), axis=1)
    cross_entropy = tf.reduce_mean(cross_entropy) * 7.
    
    # cross_entropy = slim.losses.softmax_cross_entropy(logits=saliency_map,
    #                                                   onehot_labels=binary_map)

    return cross_entropy

