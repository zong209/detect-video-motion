#! .pyenv/bin/python
# -*- encoding:utf-8 -*-
'''
@File	 : motion_detect.py
@Time	 : 2020/01/16 15:16:26
@Author   : Gaozong/260243
@Contact  : 260243@gree.com.cn/zong209@163.com
@Describe : Motion detect
'''
# import the necessary packages
import numpy as np
import imutils
import cv2
import math
from utils import image_show


def box_X_Y(box, width, height):
    """
    @Function: convert (x,y,w,h) to (minX, minY, maxX, maxY)
    @Params: 
        - box (x,y,w,h)
        - width width of image
        - height height of image
    @Result:
        (minX, minY, maxX, maxY)  float
    """
    (minX, minY) = (max(box[0] - box[2] / 2, 0), max(box[1] - box[3] / 2, 0))
    (maxX, maxY) = (min(box[0] + box[2] / 2,
                        width), min(box[1] + box[3] / 2, height))
    return (minX, minY, maxX, maxY)


def union(box1, box2, width, height):
    """
    @Function: Union two boxes
    @Params:  
        - box1 first box params (x,y,w,h)
        - box2 second box params (x,y,w,h)
        - width width of image
        - height height of image
    @Result:
        - UnionBox (x,y,w,h) int
    """
    box1 = box_X_Y(box1, width, height)
    box2 = box_X_Y(box2, width, height)
    minX = min(box1[0], box2[0])
    minY = min(box1[1], box2[1])
    maxX = max(box1[2], box2[2])
    maxY = max(box1[3], box2[3])

    x = math.ceil((minX + maxX) / 2)
    y = math.ceil((minY + maxY) / 2)
    w = maxX - minX
    h = maxY - minY
    return (int(x), int(y), int(w), int(h))


def distance(box1, box2):
    """
    @Function: Distance square bettwen box1 and box2 (center distance)
    @Params:
        - box1 first box params (x,y,w,h)
        - box2 second box params (x,y,w,h)
    @Result: Sum of distance square
    """
    return (box1[0] - box2[0])**2 + (box1[1] - box2[1])**2


def box_cluster(boxes, minDistance, height, width):
    """
    @Function: Union boxes if distance > minDistance
    @Params:
        - boxes type:list 
        - minDistance min distance (pixes)
        - width width of image
        - height height of image 
    @Result:
        - union_boxes type:list
    """
    union_boxes = []
    squa_distance = minDistance**2
    box_length = len(boxes)
    has_union = []
    for i in range(box_length):
        if i not in has_union:
            union_tmp = boxes[i]
            for j in range(i + 1, box_length):
                if distance(union_tmp, boxes[j]) < squa_distance:
                    union_tmp = union(union_tmp, boxes[j], width, height)
                    has_union.append(j)
            union_boxes.append(union_tmp)
        else:
            continue
    return union_boxes


class SingleMotionDetector:
    def __init__(self, accumWeight=0.1):
        # store the accumulated weight factor
        self.accumWeight = accumWeight

        # initialize the background model
        self.bg = None

    def update(self, image):
        # if the background model is None, initialize it
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # update the background model by accumulating the weighted
        # average
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def detect(self, image, tVal=25):
        HEIGHT, WIDTH = image.shape
        # compute the absolute difference between the background model
        # and the image passed in, then threshold the delta image
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]
        # image_show(thresh)
        # perform a series of erosions and dilations to remove small
        # blobs
        # thresh = cv2.erode(thresh, None, iterations=2)
        # image_show(thresh)
        thresh = cv2.dilate(thresh, None, iterations=2)
        # image_show(thresh)

        # find contours in the thresholded image and initialize the
        # minimum and maximum bounding box regions for motion
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # (minX, minY) = (np.inf, np.inf)
        # (maxX, maxY) = (-np.inf, -np.inf)
        # if no contours were found, return None
        if len(cnts) == 0:
            return None

        # # otherwise, loop over the contours
        cnts_array = []
        for c in cnts:
            # compute the bounding box of the contour and use it to
            # update the minimum and maximum bounding box regions
            (x, y, w, h) = cv2.boundingRect(c)

            if w > 1 and h > 1:
                cnts_array.append((x + w / 2, y + h / 2, w, h))
        cnts_union = box_cluster(cnts_array, 30, HEIGHT, WIDTH)
        #     # (minX, minY) = (min(minX, x), min(minY, y))
        #     # (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

        # otherwise, return a tuple of the thresholded image along
        # with bounding box
        return (thresh, cnts_union)
