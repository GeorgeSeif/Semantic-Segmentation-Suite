from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random
from scipy.misc import imread
import ast

# Takes an absolute file path and returns the name of the file without th extension
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


# Count total number of parameters in the model
def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

# Subtracts the mean images from ImageNet
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape exceeds image dimensions!')

# Compute the average segmentation accuracy across all classes
def compute_avg_accuracy(y_pred, y_true):
    w = y_true.shape[0]
    h = y_true.shape[1]
    total = w*h
    count = 0.0
    for i in range(w):
        for j in range(h):
            if y_pred[i, j] == y_true[i, j]:
                count = count + 1.0
    return count / total

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(y_pred, y_true, num_classes):
    w = y_true.shape[0]
    h = y_true.shape[1]
    flat_image = np.reshape(y_true, w*h)
    total = []
    for val in range(num_classes):
        total.append((flat_image == val).sum())

    count = [0.0] * num_classes
    for i in range(w):
        for j in range(h):
            if y_pred[i, j] == y_true[i, j]:
                count[int(y_pred[i, j])] = count[int(y_pred[i, j])] + 1.0

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

# Compute precision
def precision(pred, label):
    TP = np.float(np.count_nonzero(pred * label))
    FP = np.float(np.count_nonzero(pred * (label - 1)))
    prec = TP / (TP + FP)
    return prec

# Compute recall
def recall(pred, label):
    TP = np.float(np.count_nonzero(pred * label))
    FN = np.float(np.count_nonzero((pred - 1) * label))
    rec = TP / (TP + FN)
    return rec

# Compute f1 score
def f1score(pred, label):
    prec = precision(pred, label)
    rec = recall(pred, label)
    f1 = np.divide(2 * prec * rec, (prec + rec))
    return f1

def compute_mean_iou(pred, label):
    w = label.shape[0]
    h = label.shape[1]
    unique_classes = np.unique(label)
    iou_list = list([0]) * len(unique_classes)

    for index, curr_class in enumerate(unique_classes):
        pred_mask = pred[:, :] == curr_class
        label_mask = label[:, :] == curr_class

        # TP = np.float(np.count_nonzero(pred_mask * label_mask))
        # FP = np.float(np.count_nonzero(pred_mask * (label_mask - 1)))
        # FN = np.float(np.count_nonzero((pred_mask - 1) * label_mask))
        iou_and = np.float(np.sum(np.logical_and(pred_mask, label_mask)))
        iou_or = np.float(np.sum(np.logical_or(pred_mask, label_mask)))
        iou_list[index] = iou_and / iou_or

    mean_iou = np.mean(iou_list)
    return mean_iou


def evaluate_segmentation(pred, gt, num_classes):
    accuracy = compute_avg_accuracy(pred, gt)
    class_accuracies = compute_class_accuracies(pred, gt, num_classes)
    prec = precision(pred, gt)
    rec = recall(pred, gt)
    f1 = f1score(pred, gt)
    iou = compute_mean_iou(pred, gt)
    return accuracy, class_accuracies, prec, rec, f1, iou

def median_frequency_balancing(labels_dir, num_classes):
    '''
    Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c

    Where median_freq_c is the median frequency of the class for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels of the images where c appeared.

    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    #Initialize all the labels key with a list value
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

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
        j = sorted(j) #To obtain the median, we've got to sort the frequencies

        median_frequency = np.median(j) / sum(j)
        total_frequency = sum(j) / total_pixels
        median_frequency_balanced = median_frequency / total_frequency
        class_weights.append(median_frequency_balanced)


    return class_weights

# Compute the memory usage, for debugging
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)

