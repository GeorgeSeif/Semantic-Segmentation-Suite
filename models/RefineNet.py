import tensorflow as tf
from tensorflow.contrib import slim
from builders import frontend_builder
import os, sys

def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    return net

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net


def ResidualConvUnit(inputs,n_filters=256,kernel_size=3):
    """
    A local residual unit designed to fine-tune the pretrained ResNet weights

    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      kernel_size: Size of convolution kernel

    Returns:
      Output of local residual block
    """
    net=tf.nn.relu(inputs)
    net=slim.conv2d(net, n_filters, kernel_size, activation_fn=None)
    net=tf.nn.relu(net)
    net=slim.conv2d(net,n_filters,kernel_size, activation_fn=None)
    net=tf.add(net,inputs)
    return net

def ChainedResidualPooling(inputs,n_filters=256):
    """
    Chained residual pooling aims to capture background 
    context from a large image region. This component is 
    built as a chain of 2 pooling blocks, each consisting 
    of one max-pooling layer and one convolution layer. One pooling
    block takes the output of the previous pooling block as
    input. The output feature maps of all pooling blocks are 
    fused together with the input feature map through summation 
    of residual connections.

    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv

    Returns:
      Double-pooled feature maps
    """

    net_relu=tf.nn.relu(inputs)
    net=slim.max_pool2d(net_relu, [5, 5],stride=1,padding='SAME')
    net=slim.conv2d(net,n_filters,3, activation_fn=None)
    net_sum_1=tf.add(net,net_relu)

    net = slim.max_pool2d(net, [5, 5], stride=1, padding='SAME')
    net = slim.conv2d(net, n_filters, 3, activation_fn=None)
    net_sum_2=tf.add(net,net_sum_1)

    return net_sum_2


def MultiResolutionFusion(high_inputs=None,low_inputs=None,n_filters=256):
    """
    Fuse together all path inputs. This block first applies convolutions
    for input adaptation, which generate feature maps of the same feature dimension 
    (the smallest one among the inputs), and then up-samples all (smaller) feature maps to
    the largest resolution of the inputs. Finally, all features maps are fused by summation.

    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
      n_filters: Number of output feature maps for each conv

    Returns:
      Fused feature maps at higher resolution
    
    """

    if high_inputs is None: # RefineNet block 4

        fuse = slim.conv2d(low_inputs, n_filters, 3, activation_fn=None)

        return fuse

    else:

        conv_low = slim.conv2d(low_inputs, n_filters, 3, activation_fn=None)
        conv_high = slim.conv2d(high_inputs, n_filters, 3, activation_fn=None)

        conv_low_up = Upsampling(conv_low,2)

        return tf.add(conv_low_up, conv_high)


def RefineBlock(high_inputs=None,low_inputs=None):
    """
    A RefineNet Block which combines together the ResidualConvUnits,
    fuses the feature maps using MultiResolutionFusion, and then gets
    large-scale context with the ResidualConvUnit.

    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution

    Returns:
      RefineNet block for a single path i.e one resolution
    
    """

    if low_inputs is None: # block 4
        rcu_new_low= ResidualConvUnit(high_inputs, n_filters=512)
        rcu_new_low = ResidualConvUnit(rcu_new_low, n_filters=512)

        fuse = MultiResolutionFusion(high_inputs=None, low_inputs=rcu_new_low, n_filters=512)
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=512)
        output = ResidualConvUnit(fuse_pooling, n_filters=512)
        return output
    else:
        rcu_high= ResidualConvUnit(high_inputs, n_filters=256)
        rcu_high = ResidualConvUnit(rcu_high, n_filters=256)

        fuse = MultiResolutionFusion(rcu_high, low_inputs,n_filters=256)
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=256)
        output = ResidualConvUnit(fuse_pooling, n_filters=256)
        return output



def build_refinenet(inputs, num_classes, preset_model='RefineNet', frontend="ResNet101", weight_decay=1e-5, upscaling_method="bilinear", pretrained_dir="models", is_training=True):
    """
    Builds the RefineNet model. 

    Arguments:
      inputs: The input tensor
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      RefineNet model
    """

    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)

    


    high = [end_points['pool5'], end_points['pool4'],
         end_points['pool3'], end_points['pool2']]

    low = [None, None, None, None]

    # Get the feature maps to the proper size with bottleneck
    high[0]=slim.conv2d(high[0], 512, 1)
    high[1]=slim.conv2d(high[1], 256, 1)
    high[2]=slim.conv2d(high[2], 256, 1)
    high[3]=slim.conv2d(high[3], 256, 1)

    # RefineNet
    low[0]=RefineBlock(high_inputs=high[0],low_inputs=None) # Only input ResNet 1/32
    low[1]=RefineBlock(high[1],low[0]) # High input = ResNet 1/16, Low input = Previous 1/16
    low[2]=RefineBlock(high[2],low[1]) # High input = ResNet 1/8, Low input = Previous 1/8
    low[3]=RefineBlock(high[3],low[2]) # High input = ResNet 1/4, Low input = Previous 1/4

    # g[3]=Upsampling(g[3],scale=4)

    net = low[3]

    net = ResidualConvUnit(net)
    net = ResidualConvUnit(net)

    if upscaling_method.lower() == "conv":
        net = ConvUpscaleBlock(net, 128, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 128)
        net = ConvUpscaleBlock(net, 64, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 64)
    elif upscaling_method.lower() == "bilinear":
        net = Upsampling(net, scale=4)

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
