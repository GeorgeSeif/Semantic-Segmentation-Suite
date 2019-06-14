import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 2D convolution, BatchNormalization relu
	"""
	# Skip pointwise by setting num_outputs=Non
	net = slim.conv2d(inputs, n_filters, kernel_size=[1, 1], activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net

def DepthwiseSeparableConvBlock(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Builds the Depthwise Separable conv block for MobileNets
	Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
	"""
	# Skip pointwise by setting num_outputs=None
	net = slim.separable_convolution2d(inputs, num_outputs=None, depth_multiplier=1, kernel_size=[3, 3], activation_fn=None)

	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	net = slim.conv2d(net, n_filters, kernel_size=[1, 1], activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
	net = tf.nn.relu(slim.batch_norm(net))
	return net

def build_mobile_unet(inputs, preset_model, num_classes):

	has_skip = False
	if preset_model == "MobileUNet":
		has_skip = False
	elif preset_model == "MobileUNet-Skip":
		has_skip = True
	else:
		raise ValueError("Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (preset_model))

    #####################
	# Downsampling path #
	#####################
	net = ConvBlock(inputs, 64)
	net = DepthwiseSeparableConvBlock(net, 64)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_1 = net

	net = DepthwiseSeparableConvBlock(net, 128)
	net = DepthwiseSeparableConvBlock(net, 128)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_2 = net

	net = DepthwiseSeparableConvBlock(net, 256)
	net = DepthwiseSeparableConvBlock(net, 256)
	net = DepthwiseSeparableConvBlock(net, 256)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_3 = net

	net = DepthwiseSeparableConvBlock(net, 512)
	net = DepthwiseSeparableConvBlock(net, 512)
	net = DepthwiseSeparableConvBlock(net, 512)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_4 = net

	net = DepthwiseSeparableConvBlock(net, 512)
	net = DepthwiseSeparableConvBlock(net, 512)
	net = DepthwiseSeparableConvBlock(net, 512)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')


	#####################
	# Upsampling path #
	#####################
	net = conv_transpose_block(net, 512)
	net = DepthwiseSeparableConvBlock(net, 512)
	net = DepthwiseSeparableConvBlock(net, 512)
	net = DepthwiseSeparableConvBlock(net, 512)
	if has_skip:
		net = tf.add(net, skip_4)

	net = conv_transpose_block(net, 512)
	net = DepthwiseSeparableConvBlock(net, 512)
	net = DepthwiseSeparableConvBlock(net, 512)
	net = DepthwiseSeparableConvBlock(net, 256)
	if has_skip:
		net = tf.add(net, skip_3)

	net = conv_transpose_block(net, 256)
	net = DepthwiseSeparableConvBlock(net, 256)
	net = DepthwiseSeparableConvBlock(net, 256)
	net = DepthwiseSeparableConvBlock(net, 128)
	if has_skip:
		net = tf.add(net, skip_2)

	net = conv_transpose_block(net, 128)
	net = DepthwiseSeparableConvBlock(net, 128)
	net = DepthwiseSeparableConvBlock(net, 64)
	if has_skip:
		net = tf.add(net, skip_1)

	net = conv_transpose_block(net, 64)
	net = DepthwiseSeparableConvBlock(net, 64)
	net = DepthwiseSeparableConvBlock(net, 64)

	#####################
	#      Softmax      #
	#####################
	net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
	return net