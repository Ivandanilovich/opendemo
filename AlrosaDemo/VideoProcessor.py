import cv2, os


class VideoProcessor:

    @staticmethod
    def get_video_meta(id):
        vidcap = cv2.VideoCapture('../dataset/videos/{}.mp4'.format(id))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

        meta = {
            'fps': int(round(fps)),
            'duration_sec': int(round(frame_count / fps)),
            'frame_count': int(frame_count),
            'frame_width': int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'frame_height': int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            # 'guid': vidcap.get(cv2.CAP_PROP_GUID),
        }
        return meta

    @staticmethod
    def split_video_to_frames(id):
        path = '../dataset/frames/{}'.format(id)

        if os.path.isdir(path):
            return 'frames exist'

        os.mkdir(path)

        frames = []

        vidcap = cv2.VideoCapture('../dataset/videos/{}.mp4'.format(id))
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("{}/{}.jpg".format(path, count), image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
            success, image = vidcap.read()
            count += 1
        return 'frames created', frames

    @staticmethod
    def save_video_to_frames(id):
        path = '../dataset/frames/{}'.format(id)

        if os.path.isdir(path):
            return 'frames exist'

        os.mkdir(path)


        vidcap = cv2.VideoCapture('../dataset/videos/{}.mp4'.format(id))
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("{}/{}.jpg".format(path, count), image)
            success, image = vidcap.read()
            count += 1
        return 'frames created'

    @staticmethod
    def split_video(video_id, savedir=None):
        if savedir:
            print('savedir', savedir)
            os.mkdir(f'temp/{video_id}')

        frames = []

        vidcap = cv2.VideoCapture('../dataset/videos/{}.mp4'.format(video_id))
        success, image = vidcap.read()
        count = 0
        while success:
            if savedir:
                cv2.imwrite(f"temp/{video_id}/{count}.jpg", image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
            success, image = vidcap.read()
            count += 1
        return frames


if __name__ == '__main__':
    meta = VideoProcessor.get_video_meta('d9371ee6b22a44e8b164ad0c4fe2d5da')
    print(meta)
