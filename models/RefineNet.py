import tensorflow as tf
from tensorflow.contrib import slim
import resnet_v2
import os, sys

def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
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

    net = slim.max_pool2d(net_relu, [5, 5], stride=1, padding='SAME')
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

    if high_inputs is None:#refineNet block 4
        rcu_low_1 = low_inputs[0]
        rcu_low_2 = low_inputs[1]

        rcu_low_1 = slim.conv2d(rcu_low_1, n_filters, 3, activation_fn=None)
        rcu_low_2 = slim.conv2d(rcu_low_2, n_filters, 3, activation_fn=None)

        return tf.add(rcu_low_1,rcu_low_2)

    else:
        rcu_low_1 = low_inputs[0]
        rcu_low_2 = low_inputs[1]

        rcu_low_1 = slim.conv2d(rcu_low_1, n_filters, 3, activation_fn=None)
        rcu_low_2 = slim.conv2d(rcu_low_2, n_filters, 3, activation_fn=None)

        rcu_low = tf.add(rcu_low_1,rcu_low_2)

        rcu_high_1 = high_inputs[0]
        rcu_high_2 = high_inputs[1]

        rcu_high_1 = Upsampling(slim.conv2d(rcu_high_1, n_filters, 3, activation_fn=None),2)
        rcu_high_2 = Upsampling(slim.conv2d(rcu_high_2, n_filters, 3, activation_fn=None),2)

        rcu_high = tf.add(rcu_high_1,rcu_high_2)

        return tf.add(rcu_low, rcu_high)


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

    if high_inputs is None: # block 4
        rcu_low_1= ResidualConvUnit(low_inputs, n_filters=256)
        rcu_low_2 = ResidualConvUnit(low_inputs, n_filters=256)
        rcu_low = [rcu_low_1, rcu_low_2]

        fuse = MultiResolutionFusion(high_inputs=None, low_inputs=rcu_low, n_filters=256)
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=256)
        output = ResidualConvUnit(fuse_pooling, n_filters=256)
        return output
    else:
        rcu_low_1 = ResidualConvUnit(low_inputs, n_filters=256)
        rcu_low_2 = ResidualConvUnit(low_inputs, n_filters=256)
        rcu_low = [rcu_low_1, rcu_low_2]

        rcu_high_1 = ResidualConvUnit(high_inputs, n_filters=256)
        rcu_high_2 = ResidualConvUnit(high_inputs, n_filters=256)
        rcu_high = [rcu_high_1, rcu_high_2]

        fuse = MultiResolutionFusion(rcu_high, rcu_low,n_filters=256)
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=256)
        output = ResidualConvUnit(fuse_pooling, n_filters=256)
        return output



def build_refinenet(inputs, num_classes, preset_model='RefineNet-Res101', weight_decay=1e-5, is_training=True, upscaling_method="bilinear", pretrained_dir="models"):
    """
    Builds the RefineNet model. 

    Arguments:
      inputs: The input tensor
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      RefineNet model
    """

    inputs = mean_image_subtraction(inputs)

    if preset_model == 'RefineNet-Res50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training, scope='resnet_v2_50')
            # RefineNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_50.ckpt'), slim.get_model_variables('resnet_v2_50'))
    elif preset_model == 'RefineNet-Res101':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_101(inputs, is_training=is_training, scope='resnet_v2_101')
            # RefineNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_101.ckpt'), slim.get_model_variables('resnet_v2_101'))
    elif preset_model == 'RefineNet-Res152':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_152(inputs, is_training=is_training, scope='resnet_v2_152')
            # RefineNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_152.ckpt'), slim.get_model_variables('resnet_v2_152'))
    else:
    	raise ValueError("Unsupported ResNet model '%s'. This function only supports ResNet 101 and ResNet 152" % (preset_model))

    


    f = [end_points['pool5'], end_points['pool4'],
         end_points['pool3'], end_points['pool2']]

    g = [None, None, None, None]
    h = [None, None, None, None]

    for i in range(4):
        h[i]=slim.conv2d(f[i], 256, 1)

    g[0]=RefineBlock(high_inputs=None,low_inputs=h[0])
    g[1]=RefineBlock(g[0],h[1])
    g[2]=RefineBlock(g[1],h[2])
    g[3]=RefineBlock(g[2],h[3])

    # g[3]=Upsampling(g[3],scale=4)

    net = g[3]

    if upscaling_method.lower() == "conv":
        net = ConvUpscaleBlock(net, 256, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 256)
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
