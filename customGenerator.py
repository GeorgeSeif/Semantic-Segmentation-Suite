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
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map

class customGenerator:
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
        self.image_filenames = next(os.walk(image_path))[2]
        self.image_filenames.sort(key=lambda x: (len(x), x))
        self.mask_filenames = next(os.walk(mask_path))[2]
        self.mask_filenames.sort(key=lambda x: (len(x), x))

        #print("images", tuple(zip(self.image_filenames, self.mask_filenames)))

        #print("mask", self.mask_filenames)
                
        # Check image/mask pairs
        if len(self.image_filenames) != len(self.mask_filenames):
            assert ValueError('Number of images and masks does not match.')
            
        self.num_samples = len(self.image_filenames)
        
        print('Found {} samples.'.format(self.num_samples))
        
        # Random shuffle dataset        
        if shuffle:
            c = list(zip(self.image_filenames, self.mask_filenames))
            random.shuffle(c)
            self.image_filenames, self.mask_filenames = zip(*c)
            
        self.samples = itertools.cycle(zip(self.image_filenames, self.mask_filenames))


        #for i in range(5):
        #    print("SAMPLES", next(self.samples))
    
    def generator(self):
        while True:
            img_out = np.zeros(((self.batch_size,)+self.target_size+(3,)))
            mask_out = np.zeros(((self.batch_size,)+self.target_size+(self.num_class,)))
            
            for i in range(self.batch_size):
                img_file, mask_file = next(self.samples)
                img_path = os.path.join(self.image_path,img_file)
                mask_path = os.path.join(self.mask_path,mask_file)
                
                # read image and mask 
                # must convert to RGB as opencv reads w/ BGR ordering
                img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  #uint16 
                mask = cv2.cvtColor(cv2.imread(mask_path,-1), cv2.COLOR_BGR2RGB) #uint8

                #convert uint16 image to uint 8
                if img.dtype == np.uint16:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


                #stack = np.concatenate((img,mask),axis=2)
                # random flipping
                #if random.random() > 0.5:
                #    stack = np.flip(stack, 0)
                #if random.random() > 0.5:
                #    stack = np.flip(stack, 1)

                #img = stack[:,:,:3]
                #mask = stack[:,:,3:6]



                # preprocess image (convert to BGR and zero center) and mask (one hot)
                img_out[i,:,:,:] = tf.keras.applications.resnet50.preprocess_input(img)
                mask_out[i,:,:,:] = np.float32(one_hot_it(mask, self.mask_colors))
    
                # it may seem pedantic to convert to RGB only for keras preprocessing
                # to convert back to BGR, and it likely is. However, since the frontend
                # of the model is defined by tensorflow's keras implementation, I would
                # rather rely on the stability of their preprocessing, which is designed
                # to work with their model weights, than rely on my implementation of
                # the spec which may be subject to change.



                #cv2.imshow('mask', mask_out[i,:,:,:] )
                #cv2.waitKey(0)

                '''
                if len(self.aug_dict) > 0:
                    # Stack image and mask, augment together
                    im = np.concatenate((img,mask),axis=2)
                    im = augment_image(im, self.aug_dict)
                    img = im[:,:,:3]
                    mask = im[:,:,3:6]
                '''
                
                #img_out[i,:,:,:] = img.astype(np.float32) / 255.0
                
            
            # Perform random cropping
            if self.random_crop:
                xhigh = self.target_size[1]-self.random_crop[1]
                yhigh = self.target_size[0]-self.random_crop[0]
                assert (xhigh >= 0 and yhigh >= 0) # Throw error if crop is bigger than input
                x1 = 0 if xhigh == 0 else np.random.randint(0,xhigh)
                y1 = 0 if yhigh == 0 else np.random.randint(0,yhigh)
                x2 = x1 + self.random_crop[1]
                y2 = y1 + self.random_crop[0]

                #print("COORDS", x1,x2,y1,y2)
                
                img_out = img_out[:,y1:y2,x1:x2,:]
                mask_out = mask_out[:,y1:y2,x1:x2,:]






            #print("IN GENERATOR!!")
                       

            #zero centering is taken care of by keras preprocessing.
            
            # Subtract image mean for normalized float32 RGB
            #these means are from imagenet training data
            #img_out[:,:,:,0] -= 0.486
            #img_out[:,:,:,1] -= 0.456
            #img_out[:,:,:,2] -= 0.406
            
            #cv2.imshow('im', img_out[0,:,:,:] )
            #cv2.waitKey(0)
            
            #uint8 means for BGR
            #img_out[:,:,:,0] -= 103.939
            #img_out[:,:,:,1] -= 116.779
            #img_out[:,:,:,2] -= 123.68
            
            yield(img_out,mask_out)
