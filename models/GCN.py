import tensorflow as tf
from tensorflow.contrib import slim
import resnet_v2
import os, sys

def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])


def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic deconv block for GCN
    Apply Transposed Convolution for feature map upscaling
    """
    net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    return net

def BoundaryRefinementBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Boundary Refinement Block for GCN
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    net = tf.add(inputs, net)
    return net

def GlobalConvBlock(inputs, n_filters=21, size=3):
    """
    Global Conv Block for GCN
    """

    net_1 = slim.conv2d(inputs, n_filters, [size, 1], activation_fn=None, normalizer_fn=None)
    net_1 = slim.conv2d(net_1, n_filters, [1, size], activation_fn=None, normalizer_fn=None)

    net_2 = slim.conv2d(inputs, n_filters, [1, size], activation_fn=None, normalizer_fn=None)
    net_2 = slim.conv2d(net_2, n_filters, [size, 1], activation_fn=None, normalizer_fn=None)

    net = tf.add(net_1, net_2)

    return net


def build_gcn(inputs, num_classes, preset_model='GCN-Res101', weight_decay=1e-5, is_training=True, upscaling_method="bilinear", pretrained_dir="models"):
    """
    Builds the GCN model. 

    Arguments:
      inputs: The input tensor
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      GCN model
    """

    inputs = mean_image_subtraction(inputs)

    if preset_model == 'GCN-Res50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training, scope='resnet_v2_50')
            resnet_scope = 'resnet_v2_50'
            # GCN requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_50.ckpt'), slim.get_model_variables('resnet_v2_50'))
    elif preset_model == 'GCN-Res101':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_101(inputs, is_training=is_training, scope='resnet_v2_101')
            resnet_scope = 'resnet_v2_101'
            # GCN requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_101.ckpt'), slim.get_model_variables('resnet_v2_101'))
    elif preset_model == 'GCN-Res152':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_152(inputs, is_training=is_training, scope='resnet_v2_152')
            resnet_scope = 'resnet_v2_152'
            # GCN requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_152.ckpt'), slim.get_model_variables('resnet_v2_152'))
    else:
    	raise ValueError("Unsupported ResNet model '%s'. This function only supports ResNet 101 and ResNet 152" % (preset_model))

    


    res = [end_points['pool5'], end_points['pool4'],
         end_points['pool3'], end_points['pool2']]

    down_5 = GlobalConvBlock(res[0], n_filters=21, size=3)
    down_5 = BoundaryRefinementBlock(down_5, n_filters=21, kernel_size=[3, 3])
    down_5 = ConvUpscaleBlock(down_5, n_filters=21, kernel_size=[3, 3], scale=2)

    down_4 = GlobalConvBlock(res[1], n_filters=21, size=3)
    down_4 = BoundaryRefinementBlock(down_4, n_filters=21, kernel_size=[3, 3])
    down_4 = tf.add(down_4, down_5)
    down_4 = BoundaryRefinementBlock(down_4, n_filters=21, kernel_size=[3, 3])
    down_4 = ConvUpscaleBlock(down_4, n_filters=21, kernel_size=[3, 3], scale=2)

    down_3 = GlobalConvBlock(res[2], n_filters=21, size=3)
    down_3 = BoundaryRefinementBlock(down_3, n_filters=21, kernel_size=[3, 3])
    down_3 = tf.add(down_3, down_4)
    down_3 = BoundaryRefinementBlock(down_3, n_filters=21, kernel_size=[3, 3])
    down_3 = ConvUpscaleBlock(down_3, n_filters=21, kernel_size=[3, 3], scale=2)

    down_2 = GlobalConvBlock(res[3], n_filters=21, size=3)
    down_2 = BoundaryRefinementBlock(down_2, n_filters=21, kernel_size=[3, 3])
    down_2 = tf.add(down_2, down_3)
    down_2 = BoundaryRefinementBlock(down_2, n_filters=21, kernel_size=[3, 3])
    down_2 = ConvUpscaleBlock(down_2, n_filters=21, kernel_size=[3, 3], scale=2)

    net = BoundaryRefinementBlock(down_2, n_filters=21, kernel_size=[3, 3])
    net = ConvUpscaleBlock(net, n_filters=21, kernel_size=[3, 3], scale=2)
    net = BoundaryRefinementBlock(net, n_filters=21, kernel_size=[3, 3])

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net, init_fn


def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)
