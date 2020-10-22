from multiprocessing import Pool

import streamlit as st
import cv2, os, uuid, shutil
import numpy as np
import matplotlib.pyplot as plt

import sys



sys.path.append("/home/dev/alrosademo/opendemo/")
from alrosademo.utils import getframedtonohanda

from alrosademo.KeyFilter import filter_data, work_with_obs

from alrosademo.ImageProcessor import ImageProcessor
from alrosademo.LandmarkDetector import LandmarkDetector
from alrosademo.SSDDetector import SSDDetector, SSDBox

from alrosademo.VideoProcessor import VideoProcessor

st.set_option('deprecation.showfileUploaderEncoding', False)

STATIC_PATH = '/home/dev/alrosademo/anaconda3/envs/alrosa/lib/python3.7/site-packages/streamlit/static/'




imageProcessor = ImageProcessor()
ssd = SSDDetector('../models/palm_detection_builtin.tflite')
landmark = LandmarkDetector('../models/hand_landmark.tflite')


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
        'ind': int(file.split('.')[0]), 'bbox': stored_box, 'keys': stored_keys, 'handness': stored_handness,
        'handflag': stored_handflag
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

    'image meta information'
    meta

    ' '
    'splitting video into frames...'

    splitvideo_progress = st.progress(0)
    path = '../dataset/frames/{}'.format(video_id)
    os.mkdir(path)
    vidcap = cv2.VideoCapture('../dataset/videos/{}.mp4'.format(video_id))
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("{}/{}.jpg".format(path, count), image)
        success, image = vidcap.read()
        count += 1
        splitvideo_progress.progress(int(count/meta['frame_count']*100) )
    # status = VideoProcessor.save_video_to_frames(video_id)


    ' '
    'predicting...'



    files = [[f, video_id] for f in os.listdir('../dataset/frames/{}'.format(video_id))]
    downfiles = files[::2]

    my_bar = st.progress(0)
    DATA = {}

    answer=[]
    with Pool(7) as p:
        counter=0
        for x in p.imap(funtopar, downfiles):
            counter+=1
            my_bar.progress(counter/len(downfiles))
            answer.append(x)

    # with Pool(7) as p:
        # answer = p.map(funtopar, files)

    # answer
    # st.write(type(answer))
    # st.write(type(answer[0]))
    # st.write(len(answer))
    # answer

    # DATA = answer
    # import pickle
    # pickle.dump(DATA, open('../cache/{}.pickle'.format(video_id), 'wb'))
    ' '
    'visualizing...'

    DATA = {}
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

    import pickle
    pickle.dump(DATA, open('../cache/{}.pickle'.format(video_id), 'wb'))

    # frames = [plt.imread('../dataset/frames/{}/{}'.format(video_id, i)) for i in files]
    # 'frameslenhere', len(frames)
    #
    obs = filter_data(DATA, max(meta['frame_height'], meta['frame_width']))

    files = [i[0] for i in files]
    files = sorted(files, key=lambda x: int(x[:-4]))
    frames = []
    visbar = st.progress(0)
    counter = 0
    for iiii in files:
        frames.append(cv2.imread('../dataset/frames/{}/{}'.format(video_id, iiii)))
        counter+=1

        visbar.progress(counter / meta['frame_count'])

    visframesdata = work_with_obs(obs)

    for ii,jj in visframesdata:
        frames[ii] = imageProcessor.vis_hand(frames[ii], jj)



    selected_nohands = getframedtonohanda(obs, meta['fps'], meta['frame_count'])
    # selected_nohands

    for i in selected_nohands:
        # frames[i] = cv2.putText(frames[i], 'NO HANDS',
        #                         (100, 200),
        #                         cv2.FONT_HERSHEY_SIMPLEX,
        #                         1,
        #                         (255, 0, 0),
        #                         2)
        frames[i] = cv2.putText(frames[i], 'NO HANDS',
                                (0, 150),
                                cv2.FONT_HERSHEY_TRIPLEX,
                                2,
                                (0, 0, 255),
                                3)

    path_to_video = STATIC_PATH + '{}.mp4'.format(video_id + '4')
    out = cv2.VideoWriter(path_to_video,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          meta['fps'],
                          (meta['frame_width'], meta['frame_height']))
    # cv2.VideoWriter()
    ' '
    'building video...'
    buildvideo_progress = st.progress(0)
    for ind in range(len(frames)):
        out.write(frames[ind])
        buildvideo_progress.progress(int(ind / meta['frame_count'] *100) +1)
    out.release()
    st.markdown('<a href="{}"  download="video.mp4">Download video</a>'.format('{}.mp4'.format(video_id + '4')),
                unsafe_allow_html=True)


mainpipe()
