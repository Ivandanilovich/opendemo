import streamlit as st
import cv2, os, uuid, shutil
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("C:/Users/ivand/Desktop/AlrosaDemo/")

from AlrosaDemo.KeyFilter import filter_data
from AlrosaDemo.KeyFilter import work_with_obs
from AlrosaDemo.ImageProcessor import ImageProcessor
from AlrosaDemo.LandmarkDetector import LandmarkDetector
from AlrosaDemo.SSDDetector import SSDDetector, SSDBox

from AlrosaDemo.VideoProcessor import VideoProcessor

st.set_option('deprecation.showfileUploaderEncoding', False)

STATIC_PATH = 'C:/Users/ivand/.conda/envs/tf/Lib/site-packages/streamlit/static/'

def mainpipe():

    video_byteio = st.file_uploader('video')

    if video_byteio==None:
        return

    video_id = uuid.uuid4().hex
    open('../dataset/videos/{}.mp4'.format(video_id), 'wb').write(video_byteio.read())


    meta = VideoProcessor.get_video_meta(video_id)

    'image meta information', meta


    status, frames = VideoProcessor.split_video_to_frames(video_id)
    len_frames = len(frames)

    status
    ' '

    imageProcessor = ImageProcessor()
    ssd = SSDDetector('../models/palm_detection_builtin.tflite')
    landmark = LandmarkDetector('../models/hand_landmark.tflite')

    my_bar = st.progress(0)
    DATA = {}
    for mybarindex, raw_image in enumerate(frames):
        if mybarindex%2!=0:
            continue
        my_bar.progress(int(100*mybarindex/len_frames)+1)
        original_image, padded_image, norm_image, pad = imageProcessor.preprocess_image(raw_image)#NOTE CV2 FORMAT
        stored_box = []
        stored_keys = []
        stored_handness=[]; stored_handflag = []
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


        DATA[mybarindex] = {
            'bbox': stored_box, 'keys': stored_keys, 'handness': stored_handness, 'handflag': stored_handflag
        }

    import pickle
    pickle.dump(DATA, open('../cache/ti.pickle', 'wb'))
    obs = filter_data(DATA)
    frames = work_with_obs(obs, frames)


    path_to_video = STATIC_PATH+'{}.mp4'.format(video_id)
    out = cv2.VideoWriter(path_to_video,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          meta['fps'],
                          (meta['frame_width'], meta['frame_height']))
    cv2.VideoWriter()
    for ind in range(len(frames)):
        out.write(frames[ind])
    out.release()

    st.markdown('<a href="{}"  download="video.mp4">Download video</a>'.format('{}.mp4'.format(video_id)), unsafe_allow_html=True)


mainpipe()