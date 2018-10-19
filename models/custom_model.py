from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from builders import frontend_builder

def conv_block(inputs, n_filters, filter_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d(inputs, n_filters, filter_size, activation_fn=None, normalizer_fn=None)
	out = tf.nn.relu(slim.batch_norm(conv, fused=True))
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def conv_transpose_block(inputs, n_filters, strides=2, filter_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[strides, strides])
	out = tf.nn.relu(slim.batch_norm(conv, fused=True))
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def build_custom(inputs, num_classes, frontend="ResNet101", weight_decay=1e-5, is_training=True, pretrained_dir="models"):
	

	logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, is_training=is_training)

	up_1 = conv_transpose_block(end_points["pool2"], strides=4, n_filters=64)
	up_2 = conv_transpose_block(end_points["pool3"], strides=8, n_filters=64)
	up_3 = conv_transpose_block(end_points["pool4"], strides=16, n_filters=64)
	up_4 = conv_transpose_block(end_points["pool5"], strides=32, n_filters=64)

	features = tf.concat([up_1, up_2, up_3, up_4], axis=-1)

	features = conv_block(inputs=features, n_filters=256, filter_size=[1, 1])

	features = conv_block(inputs=features, n_filters=64, filter_size=[3, 3])
	features = conv_block(inputs=features, n_filters=64, filter_size=[3, 3])
	features = conv_block(inputs=features, n_filters=64, filter_size=[3, 3])


	net = slim.conv2d(features, num_classes, [1, 1], scope='logits')
	return net