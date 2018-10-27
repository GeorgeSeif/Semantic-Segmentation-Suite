# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import slim
from builders import frontend_builder
import numpy as np
import os, sys

def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    return net

def GroupedConvolutionBlock(inputs, grouped_channels, cardinality=32):
    group_list = []

    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))

    for c in range(cardinality):
        x = net[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]

        x = slim.conv2d(x, grouped_channels, kernel_size=[3, 3])

        group_list.append(x)

    group_merge = tf.concat(group_list, axis=-1)

    return group_merge

def ResNeXtBlock(inputs, n_filters_out, bottleneck_factor=2, cardinality=32):

    assert not (n_filters_out // 2) % cardinality
    grouped_channels = (n_filters_out // 2) // cardinality

    net = ConvBlock(inputs, n_filters=n_filters_out / bottleneck_factor, kernel_size=[1, 1])
    net = GroupedConvolutionBlock(net, grouped_channels, cardinality=32)
    net = ConvBlock(net, n_filters=n_filters_out, kernel_size=[1, 1])


    net = tf.add(inputs, net)

    return net

def EncoderAdaptionBlock(inputs, n_filters, bottleneck_factor=2, cardinality=32):

    net = ConvBlock(inputs, n_filters, kernel_size=[3, 3])
    net = ResNeXtBlock(net, n_filters_out=n_filters, bottleneck_factor=bottleneck_factor)
    net = ResNeXtBlock(net, n_filters_out=n_filters, bottleneck_factor=bottleneck_factor)
    net = ResNeXtBlock(net, n_filters_out=n_filters, bottleneck_factor=bottleneck_factor)
    net = ConvBlock(net, n_filters, kernel_size=[3, 3])

    return net


def SemanticFeatureGenerationBlock(inputs, D_features, D_prime_features, O_features, bottleneck_factor=2, cardinality=32):

    d_1 = ConvBlock(inputs, D_features, kernel_size=[3, 3])
    pool_1 = slim.pool(d_1, [5, 5], stride=[1, 1], pooling_type='MAX')
    d_prime_1 = ConvBlock(pool_1, D_prime_features, kernel_size=[3, 3])

    d_2 = ConvBlock(pool_1, D_features, kernel_size=[3, 3])
    pool_2 = slim.pool(d_2, [5, 5], stride=[1, 1], pooling_type='MAX')
    d_prime_2 = ConvBlock(pool_2, D_prime_features, kernel_size=[3, 3])

    d_3 = ConvBlock(pool_2, D_features, kernel_size=[3, 3])
    pool_3 = slim.pool(d_3, [5, 5], stride=[1, 1], pooling_type='MAX')
    d_prime_3 = ConvBlock(pool_3, D_prime_features, kernel_size=[3, 3])

    d_4 = ConvBlock(pool_3, D_features, kernel_size=[3, 3])
    pool_4 = slim.pool(d_4, [5, 5], stride=[1, 1], pooling_type='MAX')
    d_prime_4 = ConvBlock(pool_4, D_prime_features, kernel_size=[3, 3])


    net = tf.concat([d_prime_1, d_prime_2, d_prime_3, d_prime_4], axis=-1)

    net = ConvBlock(net, n_filters=D_features, kernel_size=[3, 3])

    net = ResNeXtBlock(net, n_filters_out=D_features, bottleneck_factor=bottleneck_factor)
    net = ResNeXtBlock(net, n_filters_out=D_features, bottleneck_factor=bottleneck_factor)
    net = ResNeXtBlock(net, n_filters_out=D_features, bottleneck_factor=bottleneck_factor)
    net = ResNeXtBlock(net, n_filters_out=D_features, bottleneck_factor=bottleneck_factor)

    net = ConvBlock(net, O_features, kernel_size=[3, 3])

    return net



def build_ddsc(inputs, num_classes, preset_model='DDSC', frontend="ResNet101", weight_decay=1e-5, is_training=True, pretrained_dir="models"):
    """
    Builds the Dense Decoder Shortcut Connections model. 

    Arguments:
      inputs: The input tensor=
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      Dense Decoder Shortcut Connections model
    """

    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)

    ### Adapting features for all stages
    decoder_4 = EncoderAdaptionBlock(end_points['pool5'], n_filters=1024)
    decoder_3 = EncoderAdaptionBlock(end_points['pool4'], n_filters=512)
    decoder_2 = EncoderAdaptionBlock(end_points['pool3'], n_filters=256)
    decoder_1 = EncoderAdaptionBlock(end_points['pool2'], n_filters=128)

    decoder_4 = SemanticFeatureGenerationBlock(decoder_4, D_features=1024, D_prime_features = 1024 / 4, O_features=1024)

    ### Fusing features from 3 and 4
    decoder_4 = ConvBlock(decoder_4, n_filters=512, kernel_size=[3, 3])
    decoder_4 = Upsampling(decoder_4, scale=2)

    decoder_3 = ConvBlock(decoder_3, n_filters=512, kernel_size=[3, 3])

    decoder_3 = tf.add_n([decoder_4, decoder_3])

    decoder_3 = SemanticFeatureGenerationBlock(decoder_3, D_features=512, D_prime_features = 512 / 4, O_features=512)

    ### Fusing features from 2, 3, 4
    decoder_4 = ConvBlock(decoder_4, n_filters=256, kernel_size=[3, 3])
    decoder_4 = Upsampling(decoder_4, scale=4)

    decoder_3 = ConvBlock(decoder_3, n_filters=256, kernel_size=[3, 3])
    decoder_3 = Upsampling(decoder_3, scale=2)

    decoder_2 = ConvBlock(decoder_2, n_filters=256, kernel_size=[3, 3])

    decoder_2 = tf.add_n([decoder_4, decoder_3, decoder_2])

    decoder_2 = SemanticFeatureGenerationBlock(decoder_2, D_features=256, D_prime_features = 256 / 4, O_features=256)

    ### Fusing features from 1, 2, 3, 4
    decoder_4 = ConvBlock(decoder_4, n_filters=128, kernel_size=[3, 3])
    decoder_4 = Upsampling(decoder_4, scale=8)

    decoder_3 = ConvBlock(decoder_3, n_filters=128, kernel_size=[3, 3])
    decoder_3 = Upsampling(decoder_3, scale=4)

    decoder_2 = ConvBlock(decoder_2, n_filters=128, kernel_size=[3, 3])
    decoder_2 = Upsampling(decoder_2, scale=2)

    decoder_1 = ConvBlock(decoder_1, n_filters=128, kernel_size=[3, 3])

    decoder_1 = tf.add_n([decoder_4, decoder_3, decoder_2, decoder_1])

    decoder_1 = SemanticFeatureGenerationBlock(decoder_1, D_features=128, D_prime_features = 128 / 4, O_features=num_classes)


    ### Final upscaling and finish
    net = Upsampling(decoder_1, scale=4)
    
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net, init_fn

