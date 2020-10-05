import math
import os

import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter

from alrosademo.ImageProcessor import ImageProcessor
from alrosademo.VideoProcessor import VideoProcessor

imageProcessor = ImageProcessor()


def filter_only2hands(handness, handflag, keys):
    handness = np.array(handness)
    handness = np.reshape(handness, (len(handness),))
    handflag = np.array(handflag)
    handflag = np.reshape(handflag, (len(handflag),))
    keys = np.array(keys)
    if len(handness)<=4:
        return handness, handflag, keys
    indexes = np.argsort(handness)
    handness = np.take(handness, indexes)[-2:]
    handflag = np.take(handflag, indexes)[-2:]
    keys = np.take(keys, indexes, 0)[-2:]
    return handness, handflag, keys

def key_center(k):
    xmin,ymin,xmax,ymax = [
            np.min(k[:,0]),
            np.min(k[:,1]),
            np.max(k[:,0]),
            np.max(k[:,1])]
    return np.array([(xmax+xmin)/2, (ymax+ymin)/2])

def distance(key_center1, key_center2):
    x1, y1 = key_center1
    x2, y2 = key_center2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def filter_data(data, image_size=('nohere')):
    print('image_size', image_size)
    DATA = []
    for frame_index, item in data.items():
        keys = item['keys']
        handness = item['handness']
        handflag = item['handflag']
        handness, handflag, keys = filter_only2hands(handness, handflag, keys)
        for n,f,k in zip(handness, handflag, keys):
            n,f,k = n,f,k
            DATA.append((frame_index, n, f, k))

    newDATA = []
    for frame_index, handness, handflag, keys in DATA:
        if handness < 0.5:
            continue
        newDATA.append((frame_index, handness, handflag, keys))

    DATA = []
    for frame_index, handness, handflag, keys in newDATA:
        xmin, ymin, xmax, ymax = [
            np.min(keys[:, 0]),
            np.min(keys[:, 1]),
            np.max(keys[:, 0]),
            np.max(keys[:, 1])]
        square = (xmax - xmin) * (ymax - ymin)
        if square < 500:
            continue
        DATA.append((frame_index, handness, handflag, keys))

    OBS = []
    for frame_index, handness, handflag, keys in DATA:
        isAdded = False
        for ob_element in OBS:
            last_one = ob_element[-1]
            delta_flag = abs(last_one[2] - handflag)
            delta_index = abs(last_one[0] - frame_index)
            delta_keys = distance(key_center(keys), key_center(last_one[3]))
            # print('delta', delta_flag, delta_index, delta_keys)
            if delta_keys < 50 \
                    and delta_flag < 0.6 \
                    and delta_index < 16 and delta_index > 0:
                ob_element.append(
                    (frame_index, handness, handflag, keys)
                )
                isAdded = True
                break

        if not isAdded:
            OBS.append([(frame_index, handness, handflag, keys)])

    return OBS


def my_interpolate(x,y):
    f = interpolate.interp1d(x, y, )
    x1 = np.arange(x[0], x[-1]+1)
    y1 = f(x1)
    if len(x1)>9:
        window_size, poly_order = 9, 2
        y1 = savgol_filter(y1, window_size, poly_order)
        return x1, y1
    if len(x1)>3:
        window_size, poly_order = 3, 2
        y1 = savgol_filter(y1, window_size, poly_order)
        return x1, y1
    return x1,y1


def work_with_obs(OBS, frames):
    obs2 = []
    for i in OBS:
        obs2.append([len(i), i])
    obs2 = sorted(obs2, key=lambda x: -x[0])

    for ob in obs2:
        if ob[0] == 1:
            continue
        x, y = [], []
        for i in ob[1]:
            # print(i[3])
            x.append(i[0])
            y.append(i[3])
        y = np.array(y)

        xses, yses = [], []
        for i in range(0, 21):
            xs, ys = y[:, i, 0], y[:, i, 1]

            newx, newxs = my_interpolate(x, xs)
            newx, newys = my_interpolate(x, ys)
            xses.append(newxs)
            yses.append(newys)

        xses = np.array(xses)
        yses = np.array(yses)
        arr = np.stack([xses, yses], 2)

        for index, i in enumerate(range(arr.shape[1])):
            key = arr[:, i, :]
            frames[newx[index]] = imageProcessor.vis_hand(frames[newx[index]], key)

    return frames

if __name__ == '__main__':
    DATA = pickle.load(open('../cache/ti.pickle', 'rb'))
    video_id = 'f1b40dd8aba94634b56b744f41fa7dca'
    meta = VideoProcessor.get_video_meta(video_id)

    newDATA = {}
    for k, item in DATA.items():
        keys = item['keys']
        handness = item['handness']
        handflag = item['handflag']
        handness, handflag, keys = filter_only2hands(handness, handflag, keys)
        newDATA[k] = {'keys': keys}

    # newDATA
    files = [f for f in os.listdir('../dataset/frames/{}'.format(video_id))]
    files = sorted(files, key=lambda x: int(x[:-4]))

    vis_images = []
    for k, item in newDATA.items():
        vis_image = plt.imread('../dataset/frames/{}/{}'.format(video_id, files[k]))
        for key in item['keys']:
            vis_image = imageProcessor.vis_hand(vis_image, key)
        vis_images.append(vis_image)

    plt.imshow(vis_images[10]);plt.show()

    path_to_video = '{}.mp4'.format(video_id)
    out = cv2.VideoWriter(path_to_video,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          meta['fps'],
                          (meta['frame_width'], meta['frame_height']))
    cv2.VideoWriter()
    for ind in range(len(vis_images)):
        out.write(vis_images[ind])
    out.release()

