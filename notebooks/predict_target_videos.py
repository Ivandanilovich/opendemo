#%%
from multiprocessing import Pool

import streamlit as st
import cv2, os, uuid, shutil
import numpy as np
import matplotlib.pyplot as plt

import sys


from alrosademo.utils import getframedtonohanda

from alrosademo.KeyFilter import filter_data, work_with_obs

from alrosademo.ImageProcessor import ImageProcessor
from alrosademo.LandmarkDetector import LandmarkDetector
from alrosademo.SSDDetector import SSDDetector, SSDBox

from alrosademo.VideoProcessor import VideoProcessor
from alrosademo.ImageProcessor import ImageProcessor
import pickle

imageProcessor = ImageProcessor()
ssd = SSDDetector('../models/palm_detection_builtin.tflite')
landmark = LandmarkDetector('../models/hand_landmark.tflite')

def funtopar(data_in):
    frame_index, image = data_in
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
        'ind': frame_index,
        'bbox': stored_box,
        'keys': stored_keys,
        'handness': stored_handness,
        'handflag': stored_handflag
    }




if __name__ == '__main__':


    path1 = 'D:/linux_share/Alrosa/selected/source/'

    vidcap = cv2.VideoCapture(path1 + 'ind_2_GH060009.MP4')
    success, image = vidcap.read()
    count = 0
    frames = []
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
        success, image = vidcap.read()
        count += 1

    with Pool(8) as p:
        answer = p.map(funtopar,
                       [[i, frame] for i, frame in enumerate(frames)]
                       )

    DATA = {}
    for i in answer:
        DATA[i['ind']] = {
            'bbox': i['bbox'],
            'keys': i['keys'],
            'handness': i['handness'],
            'handflag': i['handflag'],
        }


    pickle.dump(DATA, open('D:/linux_share/Alrosa/selected/preds/ind_2_GH060009.pickle', 'wb'))