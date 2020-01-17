#! .pyenv/bin/python
# -*- encoding:utf-8 -*-
'''
@File     : main.py
@Time     : 2020/01/16 15:24:55
@Author   : Gaozong/260243
@Contact  : 260243@gree.com.cn/zong209@163.com
@Describe : Detect difference from video
'''
import cv2
import math
import imutils
import datetime
from motion_detect import SingleMotionDetector, box_X_Y
from utils import read_video_to_frame, image_show


def detect_video_motion(video, frameCount):
    """
    @Function: Detect motion (detect difference in frames) in video
    @Params:
        -video video path or stream url
        -frameCount building background
    @Result: 
        yield frames with rectangle
    """
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    for frame in read_video_to_frame(video):
        frame = imutils.resize(frame, width=800)
        (height, width, channel) = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        # cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frameCount and total % 2 == 0:
            # detect motion in the image
            motion = md.detect(gray)

            # check to see if motion was found in the frame
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, cnts_array) = motion
                for cnt in cnts_array:
                    minX, minY, maxX, maxY = box_X_Y(cnt, width, height)

                    cv2.rectangle(frame, (math.ceil(minX), math.ceil(maxY)),
                                  (math.ceil(maxX), math.ceil(minY)),
                                  (0, 0, 255), 2)

        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1
        yield (frame.copy())


if __name__ == "__main__":
    video_path = './data/oil_show.avi'
    frame_count = 10

    for frame in detect_video_motion(video_path, frame_count):
        cv2.imshow('Detect Result', frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
