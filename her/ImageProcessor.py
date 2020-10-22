import math

import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessor:
    # @staticmethod
    def __im_normalize(self, img):
        img = (2 * ((img / 255) - 0.5)).astype('float32')
        return img

    def __pad_img(self, img):
        shape = np.array(img.shape)
        pad = (max(shape) - shape[:2]).astype('uint32') // 2
        img_pad = np.pad(img, ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
        img_pad = cv2.resize(img_pad, (img_pad.shape[0], img_pad.shape[0]))
        return img_pad, pad

    # @staticmethod
    def preprocess_image(self, image):
        padded_image, pad = self.__pad_img(image)
        # print(image.shape)
        norm_image = self.__im_normalize(cv2.resize(padded_image, (256, 256)))

        return image, padded_image, norm_image, pad

    def rotate_image(self, image, angle, image_center):
        angle = math.degrees(angle)
        image_center = tuple(image_center.astype('int'))
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    # @staticmethod
    def image_from_dir(self, path):
        image = plt.imread(path)
        return self.preprocess_image(image)

    def vis_hand(self, image, keys, color=(255,0,0)):
        def draw(p, img, index, index2):
            x1, y1 = p[index]
            x1, y1 = int(x1), int(y1)
            x2, y2 = p[index2]
            x2, y2 = int(x2), int(y2)
            return cv2.line(img, (x1, y1), (x2, y2), color, 2)

        def draw_all(img, p):
            img = draw(p, img, 20, 19)
            img = draw(p, img, 18, 19)
            img = draw(p, img, 18, 17)
            img = draw(p, img, 0, 17)
            img = draw(p, img, 0, 13)
            img = draw(p, img, 0, 9)
            img = draw(p, img, 0, 5)
            img = draw(p, img, 0, 1)
            img = draw(p, img, 2, 1)
            img = draw(p, img, 2, 3)
            img = draw(p, img, 4, 3)
            img = draw(p, img, 8, 7)
            img = draw(p, img, 6, 7)
            img = draw(p, img, 6, 5)
            img = draw(p, img, 12, 11)
            img = draw(p, img, 11, 10)
            img = draw(p, img, 10, 9)
            img = draw(p, img, 16, 15)
            img = draw(p, img, 14, 15)
            img = draw(p, img, 14, 13)
            return img

        return draw_all(image, keys)

    # def prepare_image(self, raw_image):
    #     pass


if __name__ == "__main__":
    imageProcessor = ImageProcessor()
    padded_image, norm_image, pad = imageProcessor.image_from_dir(
        '../dataset/frames/68ba5840c958490b90d7f41798ec7116/0.jpg')
    plt.imshow(norm_image);
    plt.show()
