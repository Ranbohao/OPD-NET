import numpy as np
import cv2

def binary_map_gen(img, boxes):
    temp = np.zeros([img.shape[0], img.shape[1]], np.uint8)
    boxes = boxes.astype(np.int32)
    for box in boxes:
        points = np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], np.int32)
        # print('all', points)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(temp, [points], 1)

        head_p0 = [box[0], box[1]]
        head_p1 = [box[2], box[3]]
        head_p2 = [(box[2] + box[4])/2, (box[3] + box[5])/2]
        head_p3 = [(box[0] + box[6])/2, (box[1] + box[7])/2]
        head_points = np.array([head_p0, head_p1, head_p2, head_p3], np.int32)
        # print('head', head_points)
        head_points = head_points.reshape((-1, 1, 2))
        cv2.fillPoly(temp, [head_points], 2)

    return temp

if __name__ == '__main__':
    img = np.zeros((20, 20), np.uint8)
    boxes = np.array([[0,2,2,0,10,8,8,10]])
    # print(img, boxes)
    print(binary_map_gen(img, boxes))
    # tf.image.resize_nearest_neighbor()