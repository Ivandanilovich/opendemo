import cv2
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

from alrosademo.ImageProcessor import ImageProcessor


class SSDBox:

    def calc_angle(self):
        def get_angle(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            x, y = x1 - x2, y1 - y2
            return -math.atan2(x, y)

        return get_angle(self.det['keys'][0], self.det['keys'][2])

    def getbox(self):
        return self.det['center'].copy(), self.det['size'].copy(), self.calc_angle()

    def __init__(self, det, pad, padded_image_shape):
        self.det = {
            'center': det[:2],
            'size': det[2:4],
            'keys': np.reshape(np.array(det[4:]), (-1, 2))
        }

        det = self.det.copy()
        center = det['center']
        center = center / 256 * max(padded_image_shape)
        center = np.array([center[0] - pad[1], center[1] - pad[0]])

        size = det['size']
        size = size / 256 * max(padded_image_shape)

        keys = det['keys']
        # print('keys', keys)
        # print(center, size, det['size'])
        keys[:, 0] = keys[:, 0] / det['size'][0] * size[0] + center[0]
        keys[:, 1] = keys[:, 1] / det['size'][1] * size[1] + center[1]
        # print(keys)

        self.det = {
            'center': center,
            'size': size,
            'keys': keys
        }

    def vis_on_image(self, image):
        det = self.det.copy()

        center = det['center']
        size = det['size']
        keys = det['keys']

        image = cv2.circle(image, tuple(np.round(center).astype('int')), 5, (0, 255, 0), 3)

        box = [
            center[0] - size[0] / 2,
            center[1] - size[1] / 2,
            center[0] + size[0] / 2,
            center[1] + size[1] / 2,
        ]

        image = cv2.rectangle(image,
                              tuple(np.round(box[:2]).astype('int')),
                              tuple(np.round(box[2:]).astype('int')),
                              (128, 0, 0), 2)

        for x, y in keys:
            image = cv2.circle(image, (int(round(x)), int(round(y))), 3, (255, 255, 0), 2)

        return image


# class SSDPrediction:
#
#     def __init__(self, ):


class SSDDetector:
    __anchors = np.genfromtxt('../models/anchors.csv', delimiter=',')

    def __sigm(self, x):
        y = np.array([
            1 / (1 + math.exp(-i)) for i in x
        ])
        return y

    def __non_max_suppression_fast(self, boxes, probabilities=None, overlap_threshold=0.3):

        if boxes.shape[1] == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []
        x1 = boxes[:, 0] - (boxes[:, 2] / [2])  # center x - width/2
        y1 = boxes[:, 1] - (boxes[:, 3] / [2])  # center y - height/2
        x2 = boxes[:, 0] + (boxes[:, 2] / [2])  # center x + width/2
        y2 = boxes[:, 1] + (boxes[:, 3] / [2])  # center y + height/2
        area = boxes[:, 2] * boxes[:, 3]  # width * height
        idxs = y2
        if probabilities is not None:
            idxs = probabilities
        idxs = np.argsort(idxs)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlap_threshold)[0])))
        return pick

    def __init__(self, model_path):
        self.model = tf.lite.Interpreter(model_path)
        self.model.allocate_tensors()

    def predict(self, image):
        self.model.set_tensor(0, tf.expand_dims(image, 0))
        self.model.invoke()
        out_reg = self.model.get_tensor(378)[0]
        out_clf = self.model.get_tensor(377)[0, :, 0]

        probabilities = self.__sigm(out_clf)
        detecion_mask = probabilities > 0.5
        candidate_detect = out_reg[detecion_mask]
        candidate_anchors = self.__anchors[detecion_mask]
        probabilities = probabilities[detecion_mask]
        moved_candidate_detect = candidate_detect.copy()
        moved_candidate_detect[:, :2] = candidate_detect[:, :2] + (candidate_anchors[:, :2] * 256)

        box_ids = self.__non_max_suppression_fast(moved_candidate_detect[:, :4], probabilities)
        selected = np.array([moved_candidate_detect[i] for i in box_ids])

        return selected


if __name__ == '__main__':
    ssd = SSDDetector('../models/palm_detection_builtin.tflite')

    imageProcessor = ImageProcessor()
    original_image, padded_image, norm_image, pad = imageProcessor.image_from_dir(
        'C:/Users/ivand/Desktop/AlrosaDemo/dataset/alrosa_mouth_frames/98.jpg')
    # plt.imshow(norm_image)
    # plt.show()

    d = ssd.predict(norm_image)

    box = SSDBox(d[0], pad, padded_image.shape)

    cv2.imshow('s',box.vis_on_image(original_image))
    cv2.waitKey()
    # plt.show()

    # k = box.det
    # print('angle', box.calc_angle())
    #
    # i = imageProcessor.rotate_image(
    #     box.vis_on_image(original_image),
    #     box.calc_angle(),
    #     k['center'])
    #
    # plt.imshow(i)
    # print(i)
    # plt.show()
