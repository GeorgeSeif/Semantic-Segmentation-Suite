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

def load_image(path, crop_width, crop_height):
    image = cv2.imread(path,-1)

    downscale = 1.1 * max(float(crop_height) / float(image.shape[0]), float(crop_width) / float(image.shape[1]))

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

def data_augmentation(args, input_image, output_image):
    # Data augmentation
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image

def ensure_checkpoints(model_name):
    if "Res50" in model_name and not os.path.isfile("models/resnet_v2_50.ckpt"):
        download_checkpoints("Res50")
    if "Res101" in model_name and not os.path.isfile("models/resnet_v2_101.ckpt"):
        download_checkpoints("Res101")
    if "Res152" in model_name and not os.path.isfile("models/resnet_v2_152.ckpt"):
        download_checkpoints("Res152")

def download_checkpoints(model_name):
    subprocess.check_output(["python", "get_pretrained_checkpoints.py", "--model=" + model_name])

# Get a list of the training, validation, and testing file paths
def prepare_data(dataset_dir):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        train_input_names.append(dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        train_output_names.append(dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        val_input_names.append(dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        val_output_names.append(dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        test_input_names.append(dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        test_output_names.append(dataset_dir + "/test_labels/" + file)
    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names

def build_model(model_name, net_input, num_classes, crop_width, crop_height):
    network = None
    init_fn = None
    if model_name == "FC-DenseNet56" or model_name == "FC-DenseNet67" or model_name == "FC-DenseNet103":
        network = build_fc_densenet(net_input, preset_model = model_name, num_classes=num_classes)
    elif model_name == "RefineNet-Res50" or model_name == "RefineNet-Res101" or model_name == "RefineNet-Res152":
        # RefineNet requires pre-trained ResNet weights
        network, init_fn = build_refinenet(net_input, preset_model = model_name, num_classes=num_classes)
    elif model_name == "FRRN-A" or model_name == "FRRN-B":
        network = build_frrn(net_input, preset_model = model_name, num_classes=num_classes)
    elif model_name == "Encoder-Decoder" or model_name == "Encoder-Decoder-Skip":
        network = build_encoder_decoder(net_input, preset_model = model_name, num_classes=num_classes)
    elif model_name == "MobileUNet" or model_name == "MobileUNet-Skip":
        network = build_mobile_unet(net_input, preset_model = model_name, num_classes=num_classes)
    elif model_name == "PSPNet-Res50" or model_name == "PSPNet-Res101" or model_name == "PSPNet-Res152":
        # Image size is required for PSPNet
        # PSPNet requires pre-trained ResNet weights
        network, init_fn = build_pspnet(net_input, label_size=[crop_height, crop_width], preset_model = model_name, num_classes=num_classes)
    elif model_name == "GCN-Res50" or model_name == "GCN-Res101" or model_name == "GCN-Res152":
        # GCN requires pre-trained ResNet weights
        network, init_fn = build_gcn(net_input, preset_model = model_name, num_classes=num_classes)
    elif model_name == "DeepLabV3-Res50" or model_name == "DeepLabV3-Res101" or model_name == "DeepLabV3-Res152":
        # DeepLabV requires pre-trained ResNet weights
        network, init_fn = build_deeplabv3(net_input, preset_model = model_name, num_classes=num_classes)
    elif model_name == "DeepLabV3_plus-Res50" or model_name == "DeepLabV3_plus-Res101" or model_name == "DeepLabV3_plus-Res152":
        # DeepLabV3+ requires pre-trained ResNet weights
        network, init_fn = build_deeplabv3_plus(net_input, preset_model = model_name, num_classes=num_classes)
    elif model_name == "AdapNet":
        network = build_adaptnet(net_input, num_classes=num_classes)
    elif model_name == "custom":
        network = build_custom(net_input, num_classes)
    else:
        raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

    return network, init_fn
