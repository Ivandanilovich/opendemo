import os
import shutil

import cv2
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

from alrosademo.ImageProcessor import ImageProcessor
from alrosademo.SSDDetector import SSDDetector, SSDBox
from alrosademo.VideoProcessor import VideoProcessor


class LandmarkDetector:
    def __init__(self, model_path):
        self.model = tf.lite.Interpreter(model_path)
        self.model.allocate_tensors()

    def __im_normalize(self, img):
        img = (2 * ((img / 255) - 0.5)).astype('float32')
        return img

    def predict(self, rotated_image, ssdbox):
        center, size, angle = ssdbox.getbox()
        original_center = center.copy()
        original_size = size.copy()
        center[1] -= size[1] / 2
        size *= 2.6
        cx, cy = center
        w, h = size
        xmin, ymin, xmax, ymax = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
        xmin, ymin, xmax, ymax = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))
        xmin0 = max(0, xmin)
        ymin0 = max(0, ymin)
        xmax0 = min(rotated_image.shape[1], xmax)
        ymax0 = min(rotated_image.shape[0], ymax)

        image = rotated_image[ymin0:ymax0, xmin0:xmax0]
        image = np.pad(image, ((abs(ymin0 - ymin), abs(ymax0 - ymax)), (abs(xmin0 - xmin), abs(xmax0 - xmax)), (0, 0)))

        size = image.shape[0]
        image = cv2.resize(image, (256, 256))
        image = self.__im_normalize(image)
        self.model.set_tensor(0, tf.expand_dims(tf.constant(image, tf.float32), 0))
        self.model.invoke()
        points = self.model.get_tensor(390)[0]
        handflag = self.model.get_tensor(391)[0]
        handness = self.model.get_tensor(392)[0]

        p = []
        for j in range(0, 63, 3):
            p.append([points[j], points[j + 1], points[j + 2]])
        p = np.array(p) / 256 * size

        rot_mat = np.array([
            [math.cos(-angle), -math.sin(-angle)],
            [math.sin(-angle), math.cos(-angle)]
        ])

        keys = []
        # print(original_center)
        # print(center)
        for x, y, z in p:
            # print(size_palm)
            key = np.array([
                x, y + center[1] - original_center[1]
            ])

            x, y = (np.dot(key / size - .5, rot_mat) + .5) * size - size // 2
            x += original_center[0]
            y += original_center[1]
            keys.append([x, y])

        return np.array(keys), handflag, handness

    # def subselect_image(self, rotated_image, center, size):
    #     center[1] -= size[1] / 2
    #     size *= 2.3
    #     cx, cy = center
    #     w, h = size
    #     xmin, ymin, xmax, ymax = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
    #     xmin, ymin, xmax, ymax = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))
    #     xmin0 = max(0, xmin)
    #     ymin0 = max(0, ymin)
    #     xmax0 = min(rotated_image.shape[1], xmax)
    #     ymax0 = min(rotated_image.shape[0], ymax)
    #
    #     i = rotated_image[ymin0:ymax0, xmin0:xmax0]
    #     i = np.pad(i, ((abs(ymin0 - ymin), abs(ymax0 - ymax)), (abs(xmin0 - xmin), abs(xmax0 - xmax)), (0, 0)))
    #     return i


if __name__ == '__main__':
    ssd = SSDDetector('../models/palm_detection_builtin.tflite')
    landmark = LandmarkDetector('../models/hand_landmark.tflite')
    imageProcessor = ImageProcessor()
    original_image, padded_image, norm_image, pad = imageProcessor.image_from_dir(
        'C:/Users/ivand/Desktop/1603.jpg')
    #
    # d = ssd.predict(norm_image)
    #
    # box = SSDBox(d[0], pad, padded_image.shape)

    vis_image = original_image.copy()
    for box in ssd.predict(norm_image):
        ssdbox = SSDBox(box, pad, padded_image.shape)
        ssdbox.vis_on_image(vis_image)
        k = ssdbox.det
        angle = ssdbox.calc_angle()

        rotated_image = imageProcessor.rotate_image(
            original_image,
            angle,
            k['center'].copy())

        keys, handness, handflag = landmark.predict(rotated_image, ssdbox)

        vis_image = imageProcessor.vis_hand(vis_image, keys)
        ssdbox.vis_on_image(vis_image)

    cv2.imshow('s', vis_image)
    cv2.waitKey()



    #
    # k = box.det
    # angle = box.calc_angle()
    #
    # rotated_image = imageProcessor.rotate_image(
    #     original_image,
    #     box.calc_angle(),
    #     k['center'].copy())
    #
    # # hand_image = landmark.subselect_image(rotated_image, k['center'].copy(), k['size'].copy())
    # keys, handness, handflag = landmark.predict(rotated_image, box)
    # # print(keys)
    #
    # plt.imshow(
    #     imageProcessor.vis_hand(original_image, keys))
    # plt.show()
    # imageProcessor = ImageProcessor()
    # ssd = SSDDetector('../models/palm_detection_builtin.tflite')
    # landmark = LandmarkDetector('../models/hand_landmark.tflite')
    #
    # video_id = '8e818262913c4310b7f7d294e8ccd1cd'
    # meta = VideoProcessor.get_video_meta(video_id)
    # files = [f for f in os.listdir('../dataset/frames/{}'.format(video_id))]
    # files = sorted(files, key=lambda x: int(x[:-4]))
    #
    # vis_images = []
    # for mybarindex, i in enumerate(files):
    #     if mybarindex%5!=0:
    #         continue
    #     print(mybarindex, i)
    #     original_image, padded_image, norm_image, pad = imageProcessor.image_from_dir(
    #         os.path.join('../dataset/frames/{}'.format(video_id), i))
    #     vis_image = original_image.copy()
    #
    #     for box in ssd.predict(norm_image):
    #         ssdbox = SSDBox(box, pad, padded_image.shape)
    #         k = ssdbox.det
    #         angle = ssdbox.calc_angle()
    #
    #         rotated_image = imageProcessor.rotate_image(
    #             original_image,
    #             angle,
    #             k['center'].copy())
    #
    #         keys, handness, handflag = landmark.predict(rotated_image, ssdbox)
    #
    #         vis_image = imageProcessor.vis_hand(vis_image, keys)
    #     # cv2.imshow('s' , vis_image); cv2.waitKey()
    #     vis_images.append(vis_image)
    #
    # # with st.spinner('creating video '):
    # shutil.rmtree('../dataset/vis/'+video_id)
    # os.mkdir('../dataset/vis/'+video_id)
    # for iname, i in enumerate(vis_images):
    #     plt.imsave('../dataset/vis/{}/{}.jpg'.format(video_id,iname), i)
    # files = [f for f in os.listdir('../dataset/vis/{}'.format(video_id))]
    # files = sorted(files, key=lambda x: int(x[:-4]))
    #
    #
    # frame_array=[]
    # for i in range(len(files)):
    #     filename = '../dataset/vis/{}/'.format(video_id) + files[i]
    #     print('filename', filename)
    #     img = cv2.imread(filename)
    #     height, width, layers = img.shape
    #     size = (width, height)
    #     frame_array.append(img)
    #
    # out = cv2.VideoWriter('.avi', cv2.VideoWriter_fourcc(*'DIVX'), 28, size)
    # for i in range(len(frame_array)):
    #     out.write(frame_array[i])
    # out.release()