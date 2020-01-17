#! .pyenv/bin/python
# -*- encoding:utf-8 -*-
'''
@File     : utils.py
@Time     : 2020/01/17 08:46:08
@Author   : Gaozong/260243
@Contact  : 260243@gree.com.cn/zong209@163.com
@Describe : Tool function
'''

import cv2


def read_video_to_frame(path):
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if ret:
            yield (frame)
        else:
            break


def image_show(frame):
    """
    @Function: Show image in window
    @Params: 
        -frame type: array, image 
    @Result:None
    """
    cv2.imshow('Detect Temp', frame)
    cv2.waitKey(0)