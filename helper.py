from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

from helper import *
import os

def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def one_hot_it(labels):
    w = labels.shape[0]
    h = labels.shape[1]
    x = np.zeros([w,h,12])
    for i in range(0, w):
        for j in range(0, h):
            x[i,j,labels[i][j]]=1
    return x