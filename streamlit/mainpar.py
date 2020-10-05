from multiprocessing import Pool

import streamlit as st
import cv2, os, uuid, shutil
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("C:/Users/ivand/Desktop/AlrosaDemo/")
from alrosademo.KeyFilter import filter_data, work_with_obs

from alrosademo.ImageProcessor import ImageProcessor
from alrosademo.LandmarkDetector import LandmarkDetector
from alrosademo.SSDDetector import SSDDetector, SSDBox

from alrosademo.VideoProcessor import VideoProcessor

st.set_option('deprecation.showfileUploaderEncoding', False)

STATIC_PATH = 'C:/Users/ivand/.conda/envs/tf/Lib/site-packages/streamlit/static/'


def filter_only2hands(handness, handflag, keys):
    handness = np.array(handness)
    handness = np.reshape(handness, (len(handness),))

    handflag = np.array(handflag)
    handflag = np.reshape(handflag, (len(handflag),))

    keys = np.array(keys)
    # print(keys.shape, keys)

    if len(handness) <= 2:
        return handness, handflag, keys
    indexes = np.argsort(handness)
    handness = np.take(handness, indexes)[-2:]
    handflag = np.take(handflag, indexes)[-2:]
    keys = np.take(keys, indexes, 0)[-2:]
    # keys =
    # print('sssssss')
    # print(keys.shape)
    # 7/0
    return handness, handflag, keys

imageProcessor = ImageProcessor()
ssd = SSDDetector('../models/palm_detection_builtin.tflite')
landmark = LandmarkDetector('../models/hand_landmark.tflite')
# VIDEO_ID=0
def funtopar(f):
    file, VIDEO_ID = f
    original_image, padded_image, norm_image, pad = imageProcessor.image_from_dir(
        os.path.join('../dataset/frames/{}'.format(VIDEO_ID), file))
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
        'ind': int(file.split('.')[0]), 'bbox': stored_box, 'keys': stored_keys, 'handness': stored_handness, 'handflag': stored_handflag
    }

def mainpipe():
    video_byteio = st.file_uploader('video')

    if video_byteio == None:
        return

    video_id = uuid.uuid4().hex
    # global VIDEO_ID
    # VIDEO_ID=video_id
    open('../dataset/videos/{}.mp4'.format(video_id), 'wb').write(video_byteio.read())

    meta = VideoProcessor.get_video_meta(video_id)

    'image meta information', meta

    status = VideoProcessor.save_video_to_frames(video_id)

    'frames created'
    ' '

    imageProcessor = ImageProcessor()
    ssd = SSDDetector('../models/palm_detection_builtin.tflite')
    landmark = LandmarkDetector('../models/hand_landmark.tflite')

    files = [[f,video_id] for f in os.listdir('../dataset/frames/{}'.format(video_id))]
    # files = sorted(files, key=lambda x: int(x[:-4]))
    # st.write(len(files))
    # st.write('seses')
    vis_images = []
    my_bar = st.progress(0)
    DATA = {}

    with Pool(7) as p:
        answer = p.map(funtopar, files)

    # answer
    # st.write(type(answer))
    # st.write(type(answer[0]))
    # st.write(len(answer))
    # answer

    # DATA = answer
    # import pickle
    # pickle.dump(DATA, open('../cache/{}.pickle'.format(video_id), 'wb'))
    DATA={}
    # print(answer)
    # print()
    for i in answer:

        DATA[i['ind']] = {
            'bbox': i['bbox'],
            'keys': i['keys'],
            'handness': i['handness'],
            'handflag': i['handflag'],
        }
        # 4/0

    # frames = [plt.imread('../dataset/frames/{}/{}'.format(video_id, i)) for i in files]
    # 'frameslenhere', len(frames)
    #
    obs = filter_data(DATA)
    # for i in obs:
    #     for j in i:
    #         index = j[0]
    #         key = j[3]
    #         frames[index] = imageProcessor.vis_hand(frames[index], key)
    #
    #
    #
    # path_to_video = STATIC_PATH+'{}.mp4'.format(video_id)
    # out = cv2.VideoWriter(path_to_video,
    #                       cv2.VideoWriter_fourcc(*'mp4v'),
    #                       meta['fps'],
    #                       (meta['frame_width'], meta['frame_height']))
    # cv2.VideoWriter()
    # for ind in range(len(frames)):
    #     out.write(frames[ind])
    # out.release()

    # video_file = open(STATIC_PATH+'{}.mp4'.format(video_id), 'rb')
    # video_bytes = video_file.read()
    # st.video(video_bytes)

    # st.markdown('<a href="{}"  target="_blank">Download video</a>'.format('{}.mp4'.format(video_id)), unsafe_allow_html=True)
    # st.markdown('<a href="{}"  download="video.mp4">Download video</a>'.format('{}.mp4'.format(video_id)),
    #             unsafe_allow_html=True)
    # 7/0

    frames = [cv2.imread('../dataset/frames/{}/{}'.format(video_id, i)) for i in files]
    frames = work_with_obs(obs, frames)

    path_to_video = STATIC_PATH + '{}.mp4'.format(video_id + '4')
    out = cv2.VideoWriter(path_to_video,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          meta['fps'],
                          (meta['frame_width'], meta['frame_height']))
    cv2.VideoWriter()
    for ind in range(len(frames)):
        out.write(frames[ind])
    out.release()
    st.markdown('<a href="{}"  download="video.mp4">Download video</a>'.format('{}.mp4'.format(video_id + '4')),
                unsafe_allow_html=True)


mainpipe()