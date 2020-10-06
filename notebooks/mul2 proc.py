import os
from multiprocessing import Pool
from alrosademo.ImageProcessor import ImageProcessor
from alrosademo.LandmarkDetector import LandmarkDetector
from alrosademo.SSDDetector import SSDDetector, SSDBox
from alrosademo.VideoProcessor import VideoProcessor



import time

imageProcessor = ImageProcessor()
ssd = SSDDetector('../models/palm_detection_builtin.tflite')
landmark = LandmarkDetector('../models/hand_landmark.tflite')

def funtopar(image):
    original_image, padded_image, norm_image, pad = imageProcessor.preprocess_image(image)
    stored_box = []
    stored_keys = []
    stored_handness = []
    stored_handflag = []
    for box in ssd.predict(norm_image):
        stored_box.append(box)
        ssdbox = SSDBox(box, pad, padded_image.shape)
        k = ssdbox.det
        angle = ssdbox.calc_angle()

        rotated_image = imageProcessor.rotate_image(
            original_image,
            angle,
            k['center'].copy())

        keys, handness, handflag = landmark.predict(rotated_image, ssdbox)
        stored_keys.append(keys)
        stored_handness.append(handness)
        stored_handflag.append(handflag)
    return {
        'bbox': stored_box, 'keys': stored_keys, 'handness': stored_handness,
        'handflag': stored_handflag
    }

if __name__ == '__main__':
    frames = VideoProcessor.split_videoget_frames('f781d7cc10fc41a7a24d5dcfb38e1d07')
    len(frames)


    start = time.time()






    with Pool(8) as p:
        answer = p.map(funtopar, frames)


    end = time.time()
    print(end - start)

