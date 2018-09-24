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

sys.path.append("models")
from FC_DenseNet_Tiramisu import build_fc_densenet
from Encoder_Decoder import build_encoder_decoder
from RefineNet import build_refinenet
from FRRN import build_frrn
from MobileUNet import build_mobile_unet
from PSPNet import build_pspnet
from GCN import build_gcn
from DeepLabV3 import build_deeplabv3
from DeepLabV3_plus import build_deeplabv3_plus
from AdapNet import build_adaptnet

# python looper.py \
# --model DeepLabV3_plus-Res152 \
# --input_dir /Volumes/YUGE/datasets/ade20k_floors_sss/train \
# --output_dir mloutput \
# --dataset /Volumes/YUGE/datasets/ade20k_floors_sss \
# --crop_height 512 --crop_width 512 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--class_balancing', type=str2bool, default=False, help='Whether to use median frequency class weights to balance the classes in the loss')
parser.add_argument('--input_dir', type=str, required=True, help='Directory of images to process.')
parser.add_argument('--output_dir', type=str, required=True, help='Result directory of where to place output images')
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


def load_image(path):
    image = cv2.imread(path,-1)

    downscale = 1.1 * max(float(args.crop_height) / float(image.shape[0]), float(args.crop_width) / float(image.shape[1]))

    if len(image.shape) == 3:
        shape= (int(downscale * image.shape[0]), int(downscale * image.shape[1]), 3)
    else:
        shape= (int(downscale * image.shape[0]), int(downscale * image.shape[1]))
    
    image = misc.imresize(image, shape, 'nearest')

    if (len(image.shape)<3):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def download_checkpoints(model_name):
    subprocess.check_output(["python", "get_pretrained_checkpoints.py", "--model=" + model_name])

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

if "Res50" in args.model and not os.path.isfile("models/resnet_v2_50.ckpt"):
    download_checkpoints("Res50")
if "Res101" in args.model and not os.path.isfile("models/resnet_v2_101.ckpt"):
    download_checkpoints("Res101")
if "Res152" in args.model and not os.path.isfile("models/resnet_v2_152.ckpt"):
    download_checkpoints("Res152")

# Compute your softmax cross entropy loss
print("Preparing the model ...")
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 


network = None
init_fn = None
if args.model == "FC-DenseNet56" or args.model == "FC-DenseNet67" or args.model == "FC-DenseNet103":
    network = build_fc_densenet(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "RefineNet-Res50" or args.model == "RefineNet-Res101" or args.model == "RefineNet-Res152":
    # RefineNet requires pre-trained ResNet weights
    network, init_fn = build_refinenet(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "FRRN-A" or args.model == "FRRN-B":
    network = build_frrn(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "Encoder-Decoder" or args.model == "Encoder-Decoder-Skip":
    network = build_encoder_decoder(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "MobileUNet" or args.model == "MobileUNet-Skip":
    network = build_mobile_unet(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "PSPNet-Res50" or args.model == "PSPNet-Res101" or args.model == "PSPNet-Res152":
    # Image size is required for PSPNet
    # PSPNet requires pre-trained ResNet weights
    network, init_fn = build_pspnet(net_input, label_size=[args.crop_height, args.crop_width], preset_model = args.model, num_classes=num_classes)
elif args.model == "GCN-Res50" or args.model == "GCN-Res101" or args.model == "GCN-Res152":
    # GCN requires pre-trained ResNet weights
    network, init_fn = build_gcn(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "DeepLabV3-Res50" or args.model == "DeepLabV3-Res101" or args.model == "DeepLabV3-Res152":
    # DeepLabV requires pre-trained ResNet weights
    network, init_fn = build_deeplabv3(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "DeepLabV3_plus-Res50" or args.model == "DeepLabV3_plus-Res101" or args.model == "DeepLabV3_plus-Res152":
    # DeepLabV3+ requires pre-trained ResNet weights
    network, init_fn = build_deeplabv3_plus(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "AdapNet":
    network = build_adaptnet(net_input, num_classes=num_classes)
elif args.model == "custom":
    network = build_custom(net_input, num_classes)
else:
    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")


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

image_paths = utils.get_image_paths(args.input_dir)

for path in image_paths:
    print("Processing image " + path)

    # to get the right aspect ratio of the output
    loaded_image = load_image(path)
    height, width, channels = loaded_image.shape

    resized_image = cv2.resize(loaded_image, (args.crop_width, args.crop_height))

    input_image = np.expand_dims(np.float32(resized_image),axis=0)/255.0

    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})

    run_time = time.time()-st

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)

    # this needs to get generalized
    class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    out_vis_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
    out_vis_image = cv2.resize(out_vis_image, (width, height))

    out_vis_image = cv2.addWeighted(loaded_image, 0.5, out_vis_image, 0.5,0)

    file_name = utils.filepath_to_name(path)
    output_path = os.path.join(args.output_dir, "%s_pred.png" % (file_name))
    cv2.imwrite(output_path, out_vis_image)

print("")
print("Done")

