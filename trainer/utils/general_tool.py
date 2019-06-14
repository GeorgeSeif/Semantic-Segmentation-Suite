# all general / miscellanious tools

from __future__ import print_function, division
import os, time, cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import numpy as np
import time, datetime
import os, random
from scipy.misc import imread
import ast

from utils import model_tool


"""
    display general information on the command line
    input
        args: dict of program parameters
        input_size: dict of image size
"""
def display_info(args, input_size, nb_class):

    print("\nDataset -->", args.dataset)

    if args.dataset_path != "./":
        print("Dataset path --> ", args.dataset_path)

    print("Model -->", args.model)
    print("Input Size --> %s x %s = %s" %(input_size['width'], input_size['height'], input_size['width']*input_size['height']) )
    print("Num Epochs -->", args.nb_epoch)
    print("Batch Size -->", args.batch_size)
    print("Dataset reduction factor -->", args.redux)
    print("Num Classes -->", nb_class)
    print("Learning rate -->", args.learning_rate)
    print("Regularization -->", args.regularization)
    print("Model trainable parameters -->", model_tool.count_params())

    print("Data Augmentation:")
    print("\tVertical Flip -->", args.v_flip)
    print("\tHorizontal Flip -->", args.h_flip)
    print("\tBrightness Alteration -->", args.brightness)
    print("\tRotation -->", args.rotation)
    print("")


"""
    take string as input and return boolean according to content
    input
        str
    output
        boolean
"""
def str2bool(v):

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


"""
    Takes an absolute file path and returns the name of the file without th extension
    input
        full_name
    output
        file_name
"""
def filepath_to_name(full_name):

    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]

    return file_name


# Print with time. To console or file
def LOG(X, f=None):

    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)

    
def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes) 

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

            
        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights


# Compute the memory usage, for debugging
def memory():

    import os
    import psutil

    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)

