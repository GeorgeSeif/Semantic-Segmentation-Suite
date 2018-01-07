from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools
import operator
import os

def get_class_list(list_path):
    """
    Retrieve the list of classes for the selected dataset.
    Note that the classes in the file must be LINE SEPARATED

    # Arguments
        list_path: The file path of the list of classes
        
    # Returns
        A python list of classes as strings
    """
    with open(list_path) as f:
        content = f.readlines()
    class_list = [x.strip() for x in content] 
    return class_list


def one_hot_it(label, num_classes=12):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        num_classes: The number of unique classes for this dataset
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    w = label.shape[0]
    h = label.shape[1]
    x = np.zeros([w,h,num_classes])
    unique_labels = np.unique(label)
    for i in range(0, w):
        for j in range(0, h):
            index = np.where(unique_labels==label[i][j])
            x[i,j,index]=1
    return x
    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,1])

    for i in range(0, w):
        for j in range(0, h):
            index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
            x[i, j] = index
    return x



def colour_dict(x):
    """
    Dictionairy of colour codes for visualizing segmentation results

    # Arguments
        x: Value of the current pixel

    # Returns
        Colour code
    """
    return {
        0: [64,128,64],
        1: [192,0,128],
        2: [0,128,192],
        3: [0,128,64],
        4: [128,0,0],
        5: [64,0,128],
        6: [64,0,192],
        7: [192,128,64],
        8: [192,192,128],
        9: [64,64,128],
        10: [128,0,192],
        11: [192,0,64],
        12: [128,128,64],
        13: [192,0,192],
        14: [128,64,64],
        15: [64,192,128],
        16: [64,64,0],
        17: [128,64,128],
        18: [128,128,192],
        19: [0,0,192],
        20: [192,128,128],
        21: [128,128,128],
        22: [64,128,192],
        23: [0,0,64],
        24: [0,64,64],
        25: [192,64,128],
        26: [128,128,0],
        27: [192,128,192],
        28: [64,0,64],
        29: [192,192,0],
        30: [64,192,0],
        31: [0,0,0]
    }[x]

def colour_code_segmentation(image):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        
    # Returns
        Colour coded image for segmentation visualization
    """

    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,3])
    for i in range(0, w):
        for j in range(0, h):
            x[i, j, :] = colour_dict(image[i, j, 0])
    return x


