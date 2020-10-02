import streamlit as st
import cv2, os, uuid, shutil
import numpy as np


import sys
sys.path.append("C:/Users/ivand/Desktop/AlrosaDemo/")


from AlrosaDemo.ImageProcessor import ImageProcessor
from AlrosaDemo.LandmarkDetector import LandmarkDetector
from AlrosaDemo.SSDDetector import SSDDetector, SSDBox

from AlrosaDemo.VideoProcessor import VideoProcessor

st.set_option('deprecation.showfileUploaderEncoding', False)

STATIC_PATH = 'C:/Users/ivand/.conda/envs/tf/Lib/site-packages/streamlit/static/'

btn = st.sidebar.button('clear cache')
if btn:
    shutil.rmtree('../dataset/videos')
    shutil.rmtree('../dataset/frames')
    os.mkdir('../dataset/frames')
    os.mkdir('../dataset/videos')
    st.sidebar.write('cash deleted')

def mainpipe():

    video_byteio = st.file_uploader('video .mp4 format')

    if video_byteio==None:
        return

    video_id = uuid.uuid4().hex
    open('../dataset/videos/{}.mp4'.format(video_id), 'wb').write(video_byteio.read())


    meta = VideoProcessor.get_video_meta(video_id)

    'image meta information', meta

    # with st.spinner('splitting video into frames'):
    status = VideoProcessor.split_video_to_frames(video_id)

    status
    ' '

    imageProcessor = ImageProcessor()
    ssd = SSDDetector('../models/palm_detection_builtin.tflite')
    landmark = LandmarkDetector('../models/hand_landmark.tflite')

    files = [f for f in os.listdir('../dataset/frames/{}'.format(video_id))]
    files = sorted(files, key=lambda x: int(x[:-4]))

    vis_images = []
    my_bar = st.progress(0)
    DATA = {}
    for mybarindex, i in enumerate(files):
        if mybarindex%2!=0:
            continue
        my_bar.progress(mybarindex/len(files))
        original_image, padded_image, norm_image, pad = imageProcessor.image_from_dir(os.path.join('../dataset/frames/{}'.format(video_id),i))
        vis_image = original_image.copy()
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

            vis_image = imageProcessor.vis_hand(vis_image, keys)

        DATA[mybarindex] = {
            'bbox': stored_box, 'keys': stored_keys, 'handness': stored_handness, 'handflag': stored_handflag
        }
        # cv2.imshow('s' , vis_image); cv2.waitKey()
        vis_images.append(vis_image)
    import pickle
    pickle.dump(DATA, open('../cache/{}.pickle'.format(video_id), 'wb'))

    # with st.spinner('creating video '):
    # 'len', len(vis_images)
    path_to_video = STATIC_PATH+'{}.mp4'.format(video_id)
    out = cv2.VideoWriter(path_to_video,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          meta['fps']//2,
                          (meta['frame_width'], meta['frame_height']))
    cv2.VideoWriter()
    for ind in range(len(vis_images)):
        out.write(vis_images[ind])
    out.release()

    # video_file = open(STATIC_PATH+'{}.mp4'.format(video_id), 'rb')
    # video_bytes = video_file.read()
    # st.video(video_bytes)

    # st.markdown('<a href="{}"  target="_blank">Download video</a>'.format('{}.mp4'.format(video_id)), unsafe_allow_html=True)
    st.markdown('<a href="{}"  download="video.mp4">Download video</a>'.format('{}.mp4'.format(video_id)),
                unsafe_allow_html=True)
    # 7/0




mainpipe()