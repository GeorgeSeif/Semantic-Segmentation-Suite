import cv2
import numpy as np
import itertools
import operator
import os, csv
import tensorflow as tf

import time, datetime

def get_class_dict(csv_path):
    """
    Retrieve the class dictionairy for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        A python dictionairy where the key is the class name 
        and the value is the class's pixel value
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_dict = {}
    with open(csv_path, 'rb') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_dict[row[0]] = [int(row[1]), int(row[2]), int(row[3])]
        # print(class_dict)
    return class_dict


def one_hot_it(label, class_dict):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        class_dict: A dictionairy of class--> pixel values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    unique_labels = np.array(class_dict.values())
    for colour in unique_labels:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map
    
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
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, class_dict):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        class_dict: A dictionairy of class--> pixel values
        
    # Returns
        Colour coded image for segmentation visualization
    """

    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,3])
    colour_codes = list(class_dict.values())
    for i in range(0, w):
        for j in range(0, h):
            x[i, j, :] = colour_codes[int(image[i, j])]

    return x

# class_dict = get_class_dict("CamVid/class_dict.csv")
# gt = cv2.imread("CamVid/test_labels/0001TP_007170_L.png",-1)
# gt = reverse_one_hot(one_hot_it(gt, class_dict))
# gt = colour_code_segmentation(gt, class_dict)

# file_name = "gt_test.png"
# cv2.imwrite(file_name,np.uint8(gt))