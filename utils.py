from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time
def LOG(X, f=None):
	time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	if not f:
		print(time_stamp + " " + X)
	else:
		f.write(time_stamp + " " + X)

# Replaces a select value in an array with a new value
def replace_val_in_array(input_array, original_val, replace_val = sys.maxsize - 1):
    for index, item in enumerate(input_array):
        if item == original_val:
            input_array[index] = replace_val
    return input_array

def replaces_nan_in_array(input_array, replace_val=1.0):
    for index, item in enumerate(input_array):
        if math.isnan(item):
            input_array[index] = replace_val
    return input_array


# Compute the average segmentation accuracy across all classes
def compute_avg_accuracy(y_pred, y_true):
    # print(y_true.shape)
    w = y_true.shape[0]
    h = y_true.shape[1]
    total = w*h
    count = 0.0
    for i in range(w):
        for j in range(h):
            if y_pred[i, j] == y_true[i, j]:
                count = count + 1.0
    # print(count)
    return count / total

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(y_pred, y_true, num_classes=12):
    # print(y_true.shape)
    w = y_true.shape[0]
    h = y_true.shape[1]
    flat_image = np.reshape(y_true, w*h)
    total = []
    for val in range(num_classes):
        total.append((flat_image == val).sum())

    count = [0.0] * 12
    for i in range(w):
        for j in range(h):
            if y_pred[i, j] == y_true[i, j]:
                count[int(y_pred[i, j])] = count[int(y_pred[i, j])] + 1.0
    # print(count)

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies

def precision(pred, label):
    TP = tf.count_nonzero(pred * label)
    FP = tf.count_nonzero(pred * (label - 1))

    precision = tf.divide(TP,(TP + FP))
    return precision

def recall(pred, label):
    TP = tf.count_nonzero(pred * label)
    FN = tf.count_nonzero((pred - 1) * label)
    recall = tf.divide(TP, (TP + FN))
    return recall

def f1score(pred, label):
    prec = precision(pred, label)
    rec = recall(pred, label)
    f1 = tf.divide(2 * prec * rec, (prec + rec))
    return f1

import numpy as np
import os
from scipy.misc import imread
import ast

image_dir = "./CamVid/train_labels"
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')]



def median_frequency_balancing(image_files=image_files, num_classes=12):
    '''
    Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c

    where median_freq_c is the median frequency of the class for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels of the images where c appeared.

    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately
    - num_classes(int): the number of classes of pixels in all images

    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    #Initialize all the labels key with a list value
    label_to_frequency_dict = {}
    for i in range(num_classes):
        label_to_frequency_dict[i] = []

    for n in range(len(image_files)):
        image = imread(image_files[n])
        unique_labels = list(np.unique(image))

        #For each image sum up the frequency of each label in that image and append to the dictionary if frequency is positive.
        for i in unique_labels:
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            if class_frequency != 0.0:
                index = unique_labels.index(i)
                label_to_frequency_dict[index].append(class_frequency)

    class_weights = []
    print(class_frequency)

    #Get the total pixels to calculate total_frequency later
    total_pixels = 0
    for frequencies in label_to_frequency_dict.values():
        total_pixels += sum(frequencies)

    for i, j in label_to_frequency_dict.items():
        j = sorted(j) #To obtain the median, we got to sort the frequencies

        median_frequency = np.median(j) / sum(j)
        total_frequency = sum(j) / total_pixels
        median_frequency_balanced = median_frequency / total_frequency
        class_weights.append(median_frequency_balanced)

    #Set the last class_weight to 0.0 as it's the background class
    # class_weights[-1] = 0.0

    return class_weights

if __name__ == "__main__":
    print(median_frequency_balancing(image_files, num_classes=2))

# Compute the memory usage, for debugging
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('memory use:', memoryUse)

