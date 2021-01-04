# -*- coding: utf-8 -*-
import cv2


class VideoCamera(object):
    def __init__(self):
        # self.idx = 1
        # self.limit = 1500
        raise NotImplementedError

    def get_frame(self):
        # if self.idx <= self.limit:
        #     image = cv2.imread(f'./03-sort/{self.idx:05d}.jpg')
        #     self.idx += 1
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        raise NotImplementedError
