from __future__ import print_function, division
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random
from scipy import misc
import ast
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

import helpers
import imghdr
import argparse
import glob
from PIL import Image
from scipy import misc
import fnmatch
import re
import imghdr

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getSize(filename):
    if os.path.isfile(filename): 
        st = os.stat(filename)
        return st.st_size
    else:
        return -1

def is_valid_image(path, require_rgb=True):
    try:
        what = imghdr.what(path)
        if (what != "jpeg" and what != "png"):
            return False

        im=Image.open(path)
        im.verify()
        return not require_rgb or im.mode == "RGB"
    except IOError:
        print("IOError with image " + path)
        return False
    return False

def get_image_paths(path, expression=None, filtered_dirs=None, require_rgb=True):
    file_names=[]

    # print("Checking for images at ", path, expression)

    valid_image_dir = True

    if not filtered_dirs is None and not os.path.basename(path) in filtered_dirs:
        valid_image_dir = False

    paths = os.listdir(path)

    # print("Got %d candidates" % len(paths))

    for file in paths:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            file_names.extend(get_image_paths(file_path, expression, filtered_dirs, require_rgb))
        elif valid_image_dir and is_valid_image(file_path, require_rgb) and (expression is None or fnmatch.fnmatch(file, expression)):
            file_names.append(file_path)

    file_names.sort()

    return file_names

def getRGBImage(img):
    if (len(img.shape)<3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

def getCombinedImage(a_path, b_path, a_margin=(0,0,0,0), b_margin=(0,0,0,0), a_function=None, b_function=None):

    a_image = misc.imread(a_path)
    a_image = getRGBImage(a_image)

    b_image = misc.imread(b_path)
    b_image = getRGBImage(b_image)

    ha,wa = a_image.shape[:2]
    #crop
    xa0 = a_margin[3]
    ya0 = a_margin[0]
    wa = wa - a_margin[1] - a_margin[3]
    ha = ha - a_margin[0] - a_margin[2]

    hb,wb = b_image.shape[:2]
    #crop
    xb0 = b_margin[3]
    yb0 = b_margin[0]
    wb = wb - b_margin[1] - b_margin[3]
    hb = hb - b_margin[0] - b_margin[2]

    if (ha != hb or wa != wb):
        print("A and B images must match but do not for ", a_path, b_path)
        return None

    # image[y0:y0+height , x0:x0+width, :]
    a_image = a_image[ya0:ya0+ha , xa0:xa0+wa, :]
    b_image = b_image[yb0:yb0+hb , xb0:xb0+wb, :]    

    if not a_function is None:
        a_image = a_function(a_image)

    if not b_function is None:
        b_image = b_function(b_image)

    if a_image is None or b_image is None:
        return None

    total_width = 2 * wa
    combined_img = np.zeros(shape=(ha, total_width, 3))

    combined_img[:ha,:wa]=a_image
    combined_img[:ha,wa:total_width]=b_image

    return combined_img

def hasParams(args, params):
    for param in params:
        paramValue = eval("args." + param)
        if paramValue is None:
            print("Error: argument --%s is required" % param)
            return False
    return True

def getFilteredDirs(args):
    
    filtered_dirs = None

    if not args.filter_categories is None:
        filtered_dirs = []
        if not os.path.isfile(args.filter_categories): 
                print("Error: filter_categories file %s does not exist" % args.filter_categories)
                return [], []

        with open(args.filter_categories) as f:
            content = f.readlines()
            content = [x.strip() for x in content] 

        #/b/banquet_hall 38
        for line in content:
            category_search = re.search('/[a-z]/(\\w+)', line, re.IGNORECASE)
            if category_search:
                category = category_search.group(1)
                filtered_dirs.append(category)

    return filtered_dirs

def getABImagePaths(args, require_rgb=True):
    filtered_dirs = getFilteredDirs(args)

    if not args.a_input_dir is None:
        if not hasParams(args, ["a_input_dir", "b_input_dir"]):
            return [], []

        if not os.path.isdir(args.a_input_dir): 
            print("Error: a_input_dir %s does not exist" % args.a_input_dir)
            return [], []

        if not os.path.isdir(args.b_input_dir): 
            print("Error: b_input_dir %s does not exist" % args.b_input_dir)
            return [], []

        if not filtered_dirs is None:
            filtered_dirs.append(os.path.basename(args.a_input_dir))
            filtered_dirs.append(os.path.basename(args.b_input_dir))

        a_names=get_image_paths(args.a_input_dir, args.a_match_exp, filtered_dirs=filtered_dirs, require_rgb=require_rgb)
        b_names=get_image_paths(args.b_input_dir, args.b_match_exp, filtered_dirs=filtered_dirs, require_rgb=require_rgb)
    else:

        if not hasParams(args, ["a_match_exp", "b_match_exp"]):
            return [], []

        if not os.path.isdir(args.input_dir): 
            print("Error: input_dir %s does not exist" % args.input_dir)
            return [], []

        if not filtered_dirs is None:
            filtered_dirs.append(os.path.basename(args.input_dir))

        a_names=get_image_paths(args.input_dir, args.a_match_exp, filtered_dirs=filtered_dirs, require_rgb=require_rgb)
        b_names=get_image_paths(args.input_dir, args.b_match_exp, filtered_dirs=filtered_dirs, require_rgb=require_rgb)

    return a_names, b_names



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

def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def _flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def _lovasz_softmax_flat(probas, labels, only_present=True):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.shape[1]
    losses = []
    present = []
    for c in range(C):
        fg = tf.cast(tf.equal(labels, c), probas.dtype) # foreground for class c
        if only_present:
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    losses_tensor = tf.stack(losses)
    if only_present:
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    return losses_tensor

def lovasz_softmax(probas, labels, only_present=True, per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    probas = tf.nn.softmax(probas, 3)
    labels = helpers.reverse_one_hot(labels)

    if per_image:
        def treat_image(prob, lab):
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = _flatten_probas(prob, lab, ignore, order)
            return _lovasz_softmax_flat(prob, lab, only_present=only_present)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
    else:
        losses = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore, order), only_present=only_present)
    return losses


# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    downscale = 1.5 * max(crop_height / image.shape[0], crop_width / image.shape[1])

    image = misc.imresize(image, (int(downscale * image.shape[0]), int(downscale * image.shape[1]), 3))

    if len(label.shape) == 3:
        label = misc.imresize(label, (int(downscale * label.shape[0]), int(downscale * label.shape[1]), 3))
    else:
        label = misc.imresize(label, (int(downscale * label.shape[0]), int(downscale * label.shape[1])))
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

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


def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou

    
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
        image = misc.imread(image_files[n])

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

