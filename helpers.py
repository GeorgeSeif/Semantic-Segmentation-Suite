from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools
import operator
import os

def one_hot_it(labels, num_classes=12):
    w = labels.shape[0]
    h = labels.shape[1]
    x = np.zeros([w,h,num_classes])
    unique_labels = np.unique(labels)
    for i in range(0, w):
        for j in range(0, h):
            index = np.where(unique_labels==labels[i][j])
            x[i,j,index]=1
    return x
    
def reverse_one_hot(image):
    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,1])
    # print(image.shape)
    for i in range(0, w):
        for j in range(0, h):
            index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
            x[i, j] = index
    return x



def colour_dict(x):
    return {
        0: [128,128,128],
        1: [128,0,0],
        2: [192,192,128],
        3: [128,64,128], 
        4: [60,40,222],
        5: [128,128,0],
        6: [192,128,128],
        7: [64,64,128],
        8: [64,0,128],
        9: [64,64,0],
        10: [0,128,192], 
        11: [0,0,0]
    }[x]

def colour_code_segmentation(image, dataset="CamVid"):
    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,3])
    for i in range(0, w):
        for j in range(0, h):
            x[i, j, :] = colour_dict(image[i, j, 0])
    return x

# Colour codes:
# Sky = [128,128,128]
# Building = [128,0,0]
# Pole = [192,192,128]
# Road = [128,64,128]
# Pavement = [60,40,222]
# Tree = [128,128,0]
# SignSymbol = [192,128,128]
# Fence = [64,64,128]
# Car = [64,0,128]
# Pedestrian = [64,64,0]
# Bicyclist = [0,128,192]
# Unlabelled = [0,0,0]