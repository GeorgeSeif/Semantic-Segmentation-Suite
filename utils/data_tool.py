# all tools related to the data

import cv2
import numpy as np
import itertools
import operator
import os, csv, math, sys
import tensorflow as tf
import random
from PIL import Image

import time, datetime

np.set_printoptions(threshold=sys.maxsize)


"""
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!
    input
        dataset_name: default to 'CCP' dataset
        dataset_path: folder where dataset folder is located, default to local folder
        class_file_name: name of the csv file to open
    output
        class_names_list
        label_values_list
        class_names_str
"""
def get_label_info(dataset_name="CCP", dataset_path="./", class_file_name="class_dict_colorfull.csv"):
    
    filename, file_extension = os.path.splitext( class_file_name )

    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names_list, label_values_list = [], []
    with open( dataset_path + dataset_name + "/" + class_file_name , 'r') as csvfile:

        file_reader = csv.reader( csvfile , delimiter=',')
        header = next(file_reader)

        for row in file_reader:
            class_names_list.append(row[0])
            label_values_list.append([int(float(row[1])), int(float(row[2])), int(float(row[3]))])

    class_names_str = ""
    for class_name in class_names_list:

        if not class_name == class_names_list[-1]:
            class_names_str = class_names_str + class_name + ", "
        else:
            class_names_str = class_names_str + class_name
            num_classes = len(label_values_list)

    return class_names_list, label_values_list, class_names_str


"""
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    input
        label: The 2D array segmentation image label
        label_values
        
    output
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
"""
def one_hot_it(label, label_values):
    
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
    print( len(label_values) )
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.array_equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)

    semantic_map = np.stack(semantic_map, axis=-1)
    print( semantic_map )
    # print("Time 2 = ", time.time() - st)

    return semantic_map


"""
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    input
        image: The one-hot format image 
        
    output
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class number.
"""
def reverse_one_hot(image):
    
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis = -1)

    return x


def rgb_to_onehot(rgb_arr, label_values):

    num_classes = len(label_values)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.int8 )
    #print(arr.shape)

    for i in range(num_classes):

        #print(rgb_arr.reshape( (-1,3) ))
        arr[:,:,i] = np.all( rgb_arr.reshape( (-1,3) ) == label_values[i] , axis=1).reshape( shape[:2] )
        #print(arr[:,:,i].shape)
        #input()

    return arr


def onehot_to_rgb(onehot, label_values):

    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )

    for i in range(len(label_values)):
        output[ single_layer==i ] = label_values[i]

    return np.uint8(output)


"""
    test if wether or not the provided image is RGB
    input
        image: mat
    output: True if rgb, False if grayscale
"""
def img_is_rgb(image):

    if len(image.shape) == 3:
        return True

    return False


def onehot_to_code(onehot):

    tmp = np.argmax(onehot, axis=-1)

    return tmp


"""
    Given a 1-channel array of class keys, colour code the segmentation results.
    input
        image: single channel array where each value represents the class key.
        label_values
        
    output
        Colour coded image for segmentation visualization
"""
def colour_code_segmentation(image, label_values):
    

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


"""
    apply data augmentation to input image and label image (flip, brightness, rotation)
    input
        args
        input_image
        label_image
"""
def data_augmentation(args, input_image, label_image):

    #input_image, label_image = random_crop(input_image, label_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        label_image = cv2.flip(label_image, 1)

    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        label_image = cv2.flip(label_image, 0)

    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)

    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)

    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        label_image = cv2.warpAffine(label_image, M, (label_image.shape[1], label_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, label_image


"""
    given directory of image files, return dict of file name
    input
        dataset_dir: string path to dataset dir
    output
        dataset_file_name: dict of file name
"""
def get_dataset_file_name(dataset_dir):

    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]

    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)

    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)

    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)

    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)

    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)

    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)

    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()

    dataset_file_name = {
        'training': {
            'input': train_input_names,
            'output': train_output_names},
        'validation': {
            'input': val_input_names,
            'output': val_output_names},
        'testing': {
            'input': test_input_names,
            'output': test_output_names}}

    return dataset_file_name


"""
    loop over all images and find their minimal size
    input
        dataset_dir: path to dir cointaining the dataset
    output
        tuple of minimal size coordinate
"""
def get_minimal_size( dataset_dir ):

    min_size = {'width':10000000, 'height':10000000}
    
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for name in filenames:
            try:
                width, height = Image.open(os.path.join(dirpath, name)).size
                if width < min_size['width']:
                    min_size['width'] = width
                if height < min_size['height']:
                    min_size['height'] = height
            except:
                pass

    # ensure that width and height are multiples of 8
    min_size['width'] = math.floor(min_size['width'] / 8) * 8
    min_size['height'] = math.floor(min_size['height'] / 8) * 8

    return min_size


"""
    load an image given a path
    input
        path: string
"""
def load_image(path):

    image = None
    try:
        image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    except:
        image = cv2.imread(path)

    return image


"""
    Randomly crop the image to a specific size. For data augmentation
    input
        image
        label
        crop_height
        crop_width
    output
        tuple of (image_crop, label_crop)
"""
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
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))


"""
    crop provided image and its label to the provided size
    input
        image: to crop
        label: of the image to crop as well
        min_size: dict of width and height to crop
    output
        (image_crop, label_crop) tuple
"""
def crop_image_and_label(image, label, min_size):

    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions !')

    if (min_size['width'] <= image.shape[1]) and (min_size['height'] <= image.shape[0]):

        width_border = math.ceil( ( image.shape[1] - min_size['width'] ) / 2 )
        height_border = math.ceil( ( image.shape[0] - min_size['height'] ) / 2 )
        
        if len(label.shape) == 3:

            return (
                image[height_border : height_border + min_size['height'], width_border : width_border + min_size['width'], :],
                label[height_border : height_border + min_size['height'], width_border : width_border + min_size['width'], :]
            )

        else:

            return (
                image[height_border : height_border + min_size['height'], width_border : width_border + min_size['width']],
                label[height_border : height_border + min_size['height'], width_border : width_border + min_size['width']]
            )

    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (min_size['height'], min_size['width'], image.shape[0], image.shape[1]))