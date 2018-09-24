from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess

from scipy import misc
import helpers 
import utils 

import matplotlib.pyplot as plt

import model_utils

# python looper.py \
# --model DeepLabV3_plus-Res152 \
# --input_dir uploads \
# --delete_src 1 \
# --output_dir mloutput \
# --dataset /Volumes/YUGE/datasets/ade20k_floors_sss \
# --crop_height 512 --crop_width 512 --output_color 0

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--class_balancing', type=utils.str2bool, default=False, help='Whether to use median frequency class weights to balance the classes in the loss')

parser.add_argument("--input_dir", required=False, default="uploads", help="Combined Source and Target Input Path")
parser.add_argument("--input_match_exp", required=False, help="Input Match Expression")
parser.add_argument("--filter_categories", required=False, help="Path to file with valid categories")
parser.add_argument('--output_dir', type=str, required=True, help='Result directory of where to place output images')
parser.add_argument("--output_color", type=utils.str2bool, nargs='?', const=True, default=True)
parser.add_argument("--delete_src", type=utils.str2bool, nargs='?', const=True, default=False, help="delete source images")
parser.add_argument("--run_nnet", type=utils.str2bool, nargs='?', const=True, default=True, help="Run nnet, otherwise just a filter/resize operation of images")

parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
parser.add_argument('--dataset', type=str, required=True, help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. Currently supports:\
    FC-DenseNet56, FC-DenseNet67, FC-DenseNet103, Encoder-Decoder, Encoder-Decoder-Skip, RefineNet-Res50, RefineNet-Res101, RefineNet-Res152, \
    FRRN-A, FRRN-B, MobileUNet, MobileUNet-Skip, PSPNet-Res50, PSPNet-Res101, PSPNet-Res152, GCN-Res50, GCN-Res101, GCN-Res152, DeepLabV3-Res50 \
    DeepLabV3-Res101, DeepLabV3-Res152, DeepLabV3_plus-Res50, DeepLabV3_plus-Res101, DeepLabV3_plus-Res152, AdapNet, custom')
args = parser.parse_args()

def validate_arguments(args):
    #don't waste time on invalid inputs
    if args.input_dir is None:
        print("Error: --input_dir is a required parameter for processing")
        return False
    return True

if not validate_arguments(args):
    exit()

# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
sess=tf.Session(config=config)

# Get the selected model. 
# Some of them require pre-trained ResNet
model_utils.ensure_checkpoints(args.model)

# Compute your softmax cross entropy loss
print("Preparing the model ...")
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, init_fn = model_utils.build_model(args.model, net_input, num_classes, args.crop_width, args.crop_height)

losses = None
if args.class_balancing:
    print("Computing class weights for", args.dataset, "...")
    class_weights = utils.compute_class_weights(labels_dir=args.dataset + "/train_labels", label_values=label_values)
    weights = tf.reduce_sum(class_weights * net_output, axis=-1)
    unweighted_loss = None
    unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
    losses = unweighted_loss * class_weights
else:
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
loss = tf.reduce_mean(losses)

opt = tf.train.AdamOptimizer(args.lr).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
if args.checkpoint is None:
    model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + os.path.basename(args.dataset) + ".ckpt"
else:
    model_checkpoint_name = args.checkpoint

saver.restore(sess, model_checkpoint_name)
print('Loaded latest model checkpoint')

# Load the data
print("\n***** Begin processing *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)

print("Input directory -->", args.input_dir)
print("Output directory -->", args.output_dir)
    
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# this needs to get generalized
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

while True:
    filtered_dirs = utils.getFilteredDirs(args)
    paths = utils.get_image_paths(args.input_dir, args.input_match_exp, require_rgb=False, filtered_dirs=filtered_dirs)

    num_images = len(paths)

    if num_images:
        print("Processing %d images" % num_images)
        for i in range(num_images):
            path = paths[i]
            print("Processing image " + path)

            # to get the right aspect ratio of the output
            loaded_image = model_utils.load_image(path, args.crop_width, args.crop_height)
            height, width, channels = loaded_image.shape

            resized_image = cv2.resize(loaded_image, (args.crop_width, args.crop_height))

            input_image = np.expand_dims(np.float32(resized_image),axis=0)/255.0

            st = time.time()
            output_image = sess.run(network,feed_dict={net_input:input_image})

            run_time = time.time()-st

            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            
            file_name = utils.filepath_to_name(path)

            if args.output_color:
                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                out_vis_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
                out_vis_image = cv2.resize(out_vis_image, (width, height))
                out_vis_image = cv2.addWeighted(loaded_image, 0.5, out_vis_image, 0.5,0)
                cv2.imwrite(os.path.join(args.output_dir, "%s_pred.png" % (file_name)), out_vis_image)
                

            mask = cv2.resize(np.uint8(output_image), (width, height))
            mask[mask == 0] = 255
            cv2.imwrite(os.path.join(args.output_dir, "%s_mask.png" % (file_name)), mask)

            if args.delete_src:
                os.remove(path)

        print("Waiting on images...")

    if not args.delete_src:
        print("DONE")
        exit()

    time.sleep(0.25)

