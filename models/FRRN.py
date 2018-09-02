import tensorflow as tf
from tensorflow.contrib import slim

def Upsampling(inputs,scale):
    return tf.image.resize_nearest_neighbor(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def Unpooling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ResidualUnit(inputs, n_filters=48, filter_size=3):
    """
    A local residual unit

    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      filter_size: Size of convolution kernel

    Returns:
      Output of local residual block
    """

    net = slim.conv2d(inputs, n_filters, filter_size, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, filter_size, activation_fn=None)
    net = slim.batch_norm(net, fused=True)

    return net

def FullResolutionResidualUnit(pool_stream, res_stream, n_filters_3, n_filters_1, pool_scale):
    """
    A full resolution residual unit

    Arguments:
      pool_stream: The inputs from the pooling stream
      res_stream: The inputs from the residual stream
      n_filters_3: Number of output feature maps for each 3x3 conv
      n_filters_1: Number of output feature maps for each 1x1 conv
      pool_scale: scale of the pooling layer i.e window size and stride

    Returns:
      Output of full resolution residual block
    """

    G = tf.concat([pool_stream, slim.pool(res_stream, [pool_scale, pool_scale], stride=[pool_scale, pool_scale], pooling_type='MAX')], axis=-1)

    

    net = slim.conv2d(G, n_filters_3, kernel_size=3, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters_3, kernel_size=3, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    pool_stream_out = tf.nn.relu(net)

    net = slim.conv2d(pool_stream_out, n_filters_1, kernel_size=1, activation_fn=None)
    net = Upsampling(net, scale=pool_scale)
    res_stream_out = tf.add(res_stream, net)

    return pool_stream_out, res_stream_out



def build_frrn(inputs, num_classes, preset_model='FRRN-A'):
    """
    Builds the Full Resolution Residual Network model. 

    Arguments:
      inputs: The input tensor
      preset_model: Which model you want to use. Select FRRN-A or FRRN-B
      num_classes: Number of classes

    Returns:
      FRRN model
    """

    if preset_model == 'FRRN-A':

        #####################
        # Initial Stage   
        #####################
        net = slim.conv2d(inputs, 48, kernel_size=5, activation_fn=None)
        net = slim.batch_norm(net, fused=True)
        net = tf.nn.relu(net)

        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)


        #####################
        # Downsampling Path 
        #####################
        pool_stream = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
        res_stream = slim.conv2d(net, 32, kernel_size=1, activation_fn=None)

        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX') 
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=8)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=8)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=16)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=16)

        #####################
        # Upsampling Path 
        #####################
        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=8)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=8)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)

        pool_stream = Unpooling(pool_stream, 2)

        #####################
        # Final Stage 
        #####################
        net = tf.concat([pool_stream, res_stream], axis=-1)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)

        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
        return net

        
    elif preset_model == 'FRRN-B':
        #####################
        # Initial Stage   
        #####################
        net = slim.conv2d(inputs, 48, kernel_size=5, activation_fn=None)
        net = slim.batch_norm(net, fused=True)
        net = tf.nn.relu(net)

        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)


        #####################
        # Downsampling Path 
        #####################
        pool_stream = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
        res_stream = slim.conv2d(net, 32, kernel_size=1, activation_fn=None)

        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX') 
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=8)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=8)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=16)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=16)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=32)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=32)

        #####################
        # Upsampling Path 
        #####################
        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=16)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=16)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=8)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=8)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)

        pool_stream = Unpooling(pool_stream, 2)

        #####################
        # Final Stage 
        #####################
        net = tf.concat([pool_stream, res_stream], axis=-1)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)

        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
        return net

    else:
        raise ValueError("Unsupported FRRN model '%s'. This function only supports FRRN-A and FRRN-B" % (preset_model)) 
