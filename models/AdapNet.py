# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from frontends import resnet_v2
import os, sys


def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], stride=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, n_filters, kernel_size, stride=stride, activation_fn=None, normalizer_fn=None)
    return net

def ResNetBlock_1(inputs, filters_1, filters_2):
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, filters_1, [1, 1], activation_fn=None, normalizer_fn=None)

    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    net = slim.conv2d(net, filters_1, [3, 3], activation_fn=None, normalizer_fn=None)

    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    net = slim.conv2d(net, filters_2, [1, 1], activation_fn=None, normalizer_fn=None)

    net = tf.add(inputs, net)

    return net

def ResNetBlock_2(inputs, filters_1, filters_2, s=1):
    net_1 = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net_1 = slim.conv2d(net_1, filters_1, [1, 1], stride=s, activation_fn=None, normalizer_fn=None)

    net_1 = tf.nn.relu(slim.batch_norm(net_1, fused=True))
    net_1 = slim.conv2d(net_1, filters_1, [3, 3], activation_fn=None, normalizer_fn=None)

    net_1 = tf.nn.relu(slim.batch_norm(net_1, fused=True))
    net_1 = slim.conv2d(net_1, filters_2, [1, 1], activation_fn=None, normalizer_fn=None)

    net_2 = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net_2 = slim.conv2d(net_2, filters_2, [1, 1], stride=s, activation_fn=None, normalizer_fn=None)

    net = tf.add(net_1, net_2)

    return net


def MultiscaleBlock_1(inputs, filters_1, filters_2, filters_3, p, d):
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, filters_1, [1, 1], activation_fn=None, normalizer_fn=None)

    scale_1 = tf.nn.relu(slim.batch_norm(net, fused=True))
    scale_1 = slim.conv2d(scale_1, filters_3 // 2, [3, 3], rate=p, activation_fn=None, normalizer_fn=None)
    scale_2 = tf.nn.relu(slim.batch_norm(net, fused=True))
    scale_2 = slim.conv2d(scale_2, filters_3 // 2, [3, 3], rate=d, activation_fn=None, normalizer_fn=None)
    net = tf.concat((scale_1, scale_2), axis=-1)

    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    net = slim.conv2d(net, filters_2, [1, 1], activation_fn=None, normalizer_fn=None)

    net = tf.add(inputs, net)

    return net


def MultiscaleBlock_2(inputs, filters_1, filters_2, filters_3, p, d):
    net_1 = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net_1 = slim.conv2d(net_1, filters_1, [1, 1], activation_fn=None, normalizer_fn=None)

    scale_1 = tf.nn.relu(slim.batch_norm(net_1, fused=True))
    scale_1 = slim.conv2d(scale_1, filters_3 // 2, [3, 3], rate=p, activation_fn=None, normalizer_fn=None)
    scale_2 = tf.nn.relu(slim.batch_norm(net_1, fused=True))
    scale_2 = slim.conv2d(scale_2, filters_3 // 2, [3, 3], rate=d, activation_fn=None, normalizer_fn=None)
    net_1 = tf.concat((scale_1, scale_2), axis=-1)

    net_1 = tf.nn.relu(slim.batch_norm(net_1, fused=True))
    net_1 = slim.conv2d(net_1, filters_2, [1, 1], activation_fn=None, normalizer_fn=None)

    net_2 = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net_2 = slim.conv2d(net_2, filters_2, [1, 1], activation_fn=None, normalizer_fn=None)

    net = tf.add(net_1, net_2)

    return net






def build_adaptnet(inputs, num_classes):
    """
    Builds the AdaptNet model. 

    Arguments:
      inputs: The input tensor= 
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      AdaptNet model
    """
    net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3])
    net = ConvBlock(net, n_filters=64, kernel_size=[7, 7], stride=2)
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

    net = ResNetBlock_2(net, filters_1=64, filters_2=256, s=1)
    net = ResNetBlock_1(net, filters_1=64, filters_2=256)
    net = ResNetBlock_1(net, filters_1=64, filters_2=256)

    net = ResNetBlock_2(net, filters_1=128, filters_2=512, s=2)
    net = ResNetBlock_1(net, filters_1=128, filters_2=512)
    net = ResNetBlock_1(net, filters_1=128, filters_2=512)

    skip_connection = ConvBlock(net, n_filters=12, kernel_size=[1, 1])


    net = MultiscaleBlock_1(net, filters_1=128, filters_2=512, filters_3=64, p=1, d=2)

    net = ResNetBlock_2(net, filters_1=256, filters_2=1024, s=2)
    net = ResNetBlock_1(net, filters_1=256, filters_2=1024)
    net = MultiscaleBlock_1(net, filters_1=256, filters_2=1024, filters_3=64, p=1, d=2)
    net = MultiscaleBlock_1(net, filters_1=256, filters_2=1024, filters_3=64, p=1, d=4)
    net = MultiscaleBlock_1(net, filters_1=256, filters_2=1024, filters_3=64, p=1, d=8)
    net = MultiscaleBlock_1(net, filters_1=256, filters_2=1024, filters_3=64, p=1, d=16)

    net = MultiscaleBlock_2(net, filters_1=512, filters_2=2048, filters_3=512, p=2, d=4)
    net = MultiscaleBlock_1(net, filters_1=512, filters_2=2048, filters_3=512, p=2, d=8)
    net = MultiscaleBlock_1(net, filters_1=512, filters_2=2048, filters_3=512, p=2, d=16)

    net = ConvBlock(net, n_filters=12, kernel_size=[1, 1])
    net = Upsampling(net, scale=2)

    net = tf.add(skip_connection, net)

    net = Upsampling(net, scale=8)


    
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net


def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)