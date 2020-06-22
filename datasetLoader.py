import tensorflow as tf
import numpy as np
from pathlib import Path
import os

import tensorflow_io as tfio

'''
dataset_basepath=Path("/Users/jschaffer/Semantic-Segmentation-Suite/Semantic-Segmentation-Suite/SpaceNet/")
train_images = dataset_basepath / 'train'
train_masks = dataset_basepath / 'train_labels'
val_images = dataset_basepath / 'val'
val_masks = dataset_basepath / 'val_labels'
class_dict = dataset_basepath / 'class_dict.csv'


image_path = train_images

list_ds = tf.data.Dataset.list_files(str(image_path / '*'))


def get_mask(image_path):
    parts = tf.strings.split(image_path, os.path.sep)
    print(parts)

for f in list_ds.take(5):
    print(f.numpy())
    get_mask(image_path)
'''


# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:44:48 2019

@author: Attila Lengyel

Custom image generator for image segmentation.
Generator uses BGR format.
"""

import numpy as np
import random
import os
import itertools
from PIL import Image
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K

import glob

#from scripts.transformations import augment_image

def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of num_classes
    """
 
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = K.equal(label, colour)
        class_map = K.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = K.stack(semantic_map, axis=-1)
    return semantic_map

class datasetLoader:
    def __init__(self, batch_size, image_path, mask_path, num_class,
                 target_size, aug_dict, mask_colors, shuffle = True,
                 random_crop = None):
        self.image_path = image_path
        self.mask_path = mask_path
        
        self.batch_size = batch_size
        self.num_class = num_class
        self.target_size = target_size[:2] # height, width
        self.aug_dict = aug_dict
        self.mask_colors = mask_colors
        
        self.random_crop = random_crop
        
        # Read all image and mask filenames, sort alphabetically
        #self.image_filenames = next(os.walk(image_path))[2]
        #self.image_filenames.sort()
        #self.mask_filenames = next(os.walk(mask_path))[2]
        #self.mask_filenames.sort()


        #self.image_filenames = absoluteFilePaths(self.image_path)
        #self.image_filenames.sort()
        #self.mask_filenames = absoluteFilePaths(self.mask_path)
        #self.mask_filenames.sort()

        self.image_filenames = glob.glob(str(self.image_path / '*.tif'))
        self.image_filenames.sort()
        self.mask_filenames = glob.glob(str(self.mask_path / '*.tif'))
        self.mask_filenames.sort()

                        
        # Check image/mask pairs
        if len(self.image_filenames) != len(self.mask_filenames):
            assert ValueError('Number of images and masks does not match.')
            
        self.num_samples = len(self.image_filenames)
        
        print('Found {} samples.'.format(self.num_samples))


        image_ds = tf.data.Dataset.from_tensor_slices(self.image_filenames)
        mask_ds = tf.data.Dataset.from_tensor_slices(self.mask_filenames)

        list_ds = tf.data.Dataset.zip((image_ds, mask_ds))
        
        #list_ds = tf.data.Dataset.from_tensor_slices((self.image_filenames,self.mask_filenames))
        

        def process_image_mask_pair(img_file, mask_file):

            #print("IMG_FILE", str(img_file))
            #print("Mask file", mask_file.numpy())

            #img_path = os.path.join(self.image_path,img_file)
            #mask_path = os.path.join(self.mask_path,mask_file)
            
            # Read and resize image and mask
            #img = cv2.imread(img_path, cv2.IMREAD_COLOR) # BGR
            #img = cv2.resize(img,self.target_size[::-1])
            #mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)[:,:,::-1] # BGR to RGB
            #mask = np.array(Image.fromarray(mask).resize(self.target_size[::-1], Image.NEAREST))


            img_raw = tf.io.read_file(img_file)
            #img = tf.io.decode_image(contents=img_raw, channels=3)
            img = tfio.experimental.image.decode_tiff(img_raw, index=0)

            mask_raw = tf.io.read_file(mask_file)
            #mask = tf.io.decode_image(contents=mask_raw, channels=3)
            mask = tfio.experimental.image.decode_tiff(mask_raw, index=0)

            print("mask shape", tf.keras.backend.int_shape(mask))
            print("img shape", tf.keras.backend.int_shape(img))


            #img = tf.keras.preprocessing.image.load_img(str(tf.io.decode_raw(img_file, tf.uint8)))
            #mask = tf.keras.preprocessing.image.load_img(str(tf.io.decode_raw(mask_file, tf.uint8)))


            img_out = tf.cast(img, tf.float32) / 255.0
            mask_out = (one_hot_it(mask, self.mask_colors))

            print("mask out shape", tf.keras.backend.int_shape(mask_out))
            print("img out shape", tf.keras.backend.int_shape(img_out))

            

            '''
            # Perform random cropping
            if self.random_crop:
                xhigh = self.target_size[1]-self.random_crop[1]
                yhigh = self.target_size[0]-self.random_crop[0]
                assert (xhigh >= 0 and yhigh >= 0) # Throw error if crop is bigger than input
                x1 = 0 if xhigh == 0 else np.random.randint(0,xhigh)
                y1 = 0 if yhigh == 0 else np.random.randint(0,yhigh)
                x2 = x1 + self.random_crop[1]
                y2 = y1 + self.random_crop[0]

                print("COORDS", x1,x2,y1,y2)
                
                img_out = img_out[y1:y2,x1:x2,:]
                mask_out = mask_out[y1:y2,x1:x2,:]

                img_out = tf.reshape(img_out, shape=self.random_crop)
            '''
            img_out= tf.image.random_crop(img_out, self.random_crop, seed=1)
            mask_out= tf.image.random_crop(mask_out, (448,448,2), seed=1)
            return img_out, mask_out


        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.labeled_ds = list_ds.map(process_image_mask_pair, num_parallel_calls=self.AUTOTUNE)

       


        '''
        zip
        load images
        transform



        prepare for trianing

        -repeat
        -shuffle
        -batch
        -prefetch
        '''

    def prepare_for_training(self, cache=True, shuffle_buffer_size=1000):

        ds = self.labeled_ds

        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        ds =ds.repeat()

        ds =ds.batch(self.batch_size)


        #ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

    
    





