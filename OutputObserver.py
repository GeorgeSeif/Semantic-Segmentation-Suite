# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:00:29 2019
@author: beheerder
"""

import cv2
import os
import io
import numpy as np
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import tensorflow as tf

class OutputObserver(Callback):
    """"
    Callback to save segmentation predictions during training.
    
    # Arguments:
        data            data that should be used for prediction
        output_path     directory where epoch predictions are to be stored
        mask_colors     class colors used for visualizing predictions
        batch_size      batch size used for prediction, default 2
        tmp_interval    save interval for tmp image in batches, default 100
    """
    def __init__(self, data, output_path, mask_colors, batch_size = 1, tmp_interval = 100):
        self.epoch = 0
        self.input = data[0] #source image
        self.target = data[1] #segmented image
        self.output_path = output_path
        self.mask_colors = mask_colors
        
        self.tmp_interval = tmp_interval
        self.batch_size = batch_size

        #logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # Creates a file writer for the log directory.
        self.file_writer = tf.summary.create_file_writer(output_path)


        '''
        if isinstance(data,(list,)):
            data_len = data[0].shape[0]
            data_out = data[0]
        else:
            data_len = data.shape[0]
            data_out = data
        
        self.batch_size = np.minimum(batch_size,data_len)
        self.tmp_interval = tmp_interval

        # Save input files
        data_out[:,:,:,0] += 103.939
        data_out[:,:,:,1] += 116.779
        data_out[:,:,:,2] += 123.68
        data_out = data_out.astype('uint8')
        if data_out.shape[-1] == 2:
            data_out = np.concatenate((data_out, np.zeros((data_out.shape[:3]+(1,)))), axis=-1)
        elif data_out.shape[-1] == 1:
            data_out = np.concatenate((data_out, np.zeros((data_out.shape[:3]+(2,)))), axis=-1)
        for i in range(data_out.shape[0]):
            cv2.imwrite(os.path.join(self.output_path,'input_{}.png'.format(i)),data_out[i,:,:,:])
        '''


    def labelVisualize(self, y_pred):
        """
        Convert prediction to color-coded image.
        """
        x = np.argmax(y_pred, axis=-1)
        colour_codes = np.array(self.mask_colors)
        img = colour_codes[x.astype('uint8')]
        return img
        
    #def on_train_begin(self, logs={}):
    #    y_pred = self.model.predict(self.data, batch_size=self.batch_size)
    #    img = self.labelVisualize(y_pred[0,:,:,:])
    #    cv2.imwrite(os.path.join(self.output_path,'init.png'),img[:,:,::-1])


    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def get_figure(self, display_list):
        figure = plt.figure(figsize=(15, 15))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        #plt.show()

        return figure

    def on_batch_end(self, batch, logs={}):
        if batch % self.tmp_interval == 0:
            y_pred = self.model.predict(self.input, batch_size=self.batch_size)
            #np.save(os.path.join(self.output_path,'tmp.npy'),y_pred)
            img = self.labelVisualize(y_pred[0,:,:,:])
            #print("img_shape", tf.keras.backend.int_shape(img))

            #print("input_shape",  tf.keras.backend.int_shape(self.input[0]))
            #print("target_shape",  tf.keras.backend.int_shape(self.target[0]))
            #cv2.imwrite(os.path.join(self.output_path,'tmp.png'),img[:,:,::-1])
            target_labelled = self.labelVisualize(self.target[0])
            
            
            #self.display([self.input[0], target_labelled, img]) #self.target[0],
            fig = self.get_figure([self.input[0], target_labelled, img])

            with self.file_writer.as_default():
                tf.summary.image("Training data", self.plot_to_image(fig), step=(100000*self.epoch+batch)) #step=batch

            
    
    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
    #    y_pred = self.model.predict(self.data, batch_size=self.batch_size)
    #    np.save(os.path.join(self.output_path,'epoch_{}_img.npy'.format(epoch)),y_pred)
    #    for i in range(y_pred.shape[0]):
    #        img = self.labelVisualize(y_pred[i,:,:,:])
    #        cv2.imwrite(os.path.join(self.output_path,'epoch_{}_img_{}.png'.format(epoch,i)),img[:,:,::-1])