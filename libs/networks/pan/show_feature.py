import tensorflow as tf
import numpy as np
import cv2

def feature_map_normalize(image, channel='RGB'):

    def np_normalize(img):
        img = np.array(img * 255)
        img = np.clip(img, 0, 255)
        img = np.array(img, np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        if channel == 'RGB':
            img = np.stack((img[:,:,2], img[:,:,1], img[:,:,0]), axis=2)
        return img
    image = tf.squeeze(image, 0) #[w / S, h / S, 1]
    image = tf.py_func(np_normalize, inp=[image], Tout=[tf.uint8]) #[w / S, h / S, 3]
    image = tf.expand_dims(image, 0) #[1, w / S, h / S, 3]
    
    return image


def feature_map_process_cv(image):

    image = np.array(image * 255)
    image = np.clip(image, 0, 255)
    image = np.array(image,np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image

