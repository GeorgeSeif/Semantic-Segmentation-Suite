import tensorflow as tf
from tensorflow.contrib import slim
import math
import resnet_v1

def Upsampling(inputs,feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=[feature_map_shape[1], feature_map_shape[2]])

def InterpBlock(net, level, feature_map_shape, pooling_type):
    kernel_strides_map = {1: 60,
                          2: 30,
                          3: 20,
                          6: 10}
    
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])

    net = slim.pool(net, kernel, stride=strides, pooling_type='MAX')
    net = slim.conv2d(net, 512, [1, 1], activation_fn=None)
    net = slim.batch_norm(net)
    net = tf.nn.relu(net)
    net = Upsampling(net, feature_map_shape)
    return net

def PyramidPoolingModule(inputs, pooling_type):
    """
    Build the Pyramid Pooling Module.
    """

    feature_map_size = tf.cast(tf.cast(tf.shape(inputs), tf.float32) / tf.convert_to_tensor(8.0), tf.int32)

    interp_block1 = InterpBlock(inputs, 1, feature_map_size, pooling_type)
    interp_block2 = InterpBlock(inputs, 2, feature_map_size, pooling_type)
    interp_block3 = InterpBlock(inputs, 3, feature_map_size, pooling_type)
    interp_block6 = InterpBlock(inputs, 6, feature_map_size, pooling_type)

    res = tf.concat([inputs, interp_block6, interp_block3, interp_block2, interp_block1], axis=-1)
    return res



def build_pspnet(inputs, preset_model='PSPNet-Res50', pooling_type = "MAX", num_classes=12, weight_decay=1e-5, is_training=True):
    """
    Builds the PSPNet model. 

    Arguments:
      inputs: The input tensor
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes
      pooling_type: Max or Average pooling

    Returns:
      PSPNet model
    """

    inputs = mean_image_subtraction(inputs)

    if preset_model == 'PSPNet-Res50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_50(inputs, is_training=is_training, scope='resnet_v1_101')
    elif preset_model == 'PSPNet-Res101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_101(inputs, is_training=is_training, scope='resnet_v1_101')
    elif preset_model == 'PSPNet-Res152':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_152(inputs, is_training=is_training, scope='resnet_v1_152')
    else:
        raise ValueError("Unsupported ResNet model '%s'. This function only supports ResNet 50, ResNet 101, and ResNet 152" % (preset_model))

    


    f = [end_points['pool5'], end_points['pool4'],
         end_points['pool3'], end_points['pool2']]


    original_shape = tf.shape(inputs)
    psp = PyramidPoolingModule(f[0], pooling_type=pooling_type)

    net = slim.conv2d(psp, 512, [3, 3], activation_fn=None)
    net = slim.batch_norm(net)
    net = tf.nn.relu(net)
    
    net = slim.dropout(net, keep_prob=(0.9))
    
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    net = Upsampling(net, original_shape)

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