import re



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPool2D, UpSampling2D, BatchNormalization, ZeroPadding2D, Layer

from tensorflow.keras.applications import ResNet50, ResNet101

#from models.ResNet_101 import resnet101_model

kern_init = keras.initializers.he_normal()
kern_reg = keras.regularizers.l2(1e-5)


class ScaledAdd(Layer):
  '''
  Upscales the lower of two tensors and then adds them elementwise.
  '''

  def __init__(self):
    super(ScaledAdd, self).__init__()

  def build(self, input_shape):
    # low_size = keras.backend.int_shape(conv_low)[1:3]
    # high_size = keras.backend.int_shape(conv_high)[1:3]

    high_shape = input_shape[0]
    low_shape = input_shape[1]

    if high_shape is not None and low_shape is not None:
      high_dim = high_shape[1:3]
      low_dim = low_shape[1:3]
      self.size = (high_dim[0]/low_dim[0], high_dim[1]/low_dim[1])

  def call(self, high, low):

    low_dim = keras.backend.int_shape(low)[1:3]
    high_dim = keras.backend.int_shape(high)[1:3]
    size = (high_dim[0]/low_dim[0], high_dim[1]/low_dim[1])
    low_up = UpSampling2D(size=size, interpolation='bilinear')(low)
    out = Add()([high, low_up])
    return out


def ResidualConvUnit(inputs, n_filters=256, kernel_size=3, name=''):
    """
    A local residual unit designed to fine-tune the pretrained ResNet weights

    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      kernel_size: Size of convolution kernel

    Returns:
      Output of local residual block
    """

    net = ReLU(name=name+"relu1")(inputs)
    net = Conv2D(n_filters, kernel_size, padding="same",  name=name+'conv1',
                 kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = ReLU(name=name+"relu2")(net)
    net = Conv2D(n_filters, kernel_size, padding="same",  name=name+'conv2',
                 kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)

    # print("net", net)
    # print("inputs", inputs)

    net = Add(name=name+'sum')([net, inputs])

    return net


'''def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, n_filters, kernel_size,
                      activation_fn=None, normalizer_fn=None)
    return net'''


def BilinearUpsampling(inputs, scale):
    return tf.image.resize(inputs, size=[tf.shape(input=inputs)[1]*scale,  tf.shape(input=inputs)[2]*scale], method=tf.image.ResizeMethod.BILINEAR)


def ChainedResidualPooling(inputs, n_filters=256, name=''):
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
    '''
    # using the 4 layer pooling claimed in paper as better for segmentation tasks
    net_relu = ReLU(name=name+'relu')(inputs)

    net = Conv2D(n_filters, 3, padding='same', name=name+'conv1',
                 kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net_relu)
    net = BatchNormalization()(net)
    net = MaxPool2D([5, 5], strides=1,  name=name+'pool1', padding='SAME')(net)
    net_out_1 = net

    net = Conv2D(n_filters, 3, padding='same', name=name+'conv2',
                 kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D([5, 5], strides=1,  name=name+'pool2', padding='SAME')(net)
    net_out_2 = net

    net = Conv2D(n_filters, 3, padding='same', name=name+'conv3',
                 kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D([5, 5], strides=1,  name=name+'pool3', padding='SAME')(net)
    net_out_3 = net

    net = Conv2D(n_filters, 3, padding='same', name=name+'conv4',
                 kernel_initializer=kern_init, kernel_regularizer=kern_reg)(net)
    net = BatchNormalization()(net)
    net = MaxPool2D([5, 5], strides=1,  name=name+'pool4', padding='SAME')(net)
    net_out_4 = net

    net_sum_2 = Add(
        name=name+'sum')([net_relu, net_out_1, net_out_2, net_out_3, net_out_4])

    return net_sum_2
    '''

    net_relu = ReLU(name=name+'relu')(inputs)


    net = MaxPool2D([5, 5], strides=1,  name=name+ \
                    'pool1', padding='SAME')(net_relu)
    net = Conv2D(n_filters, 3, name=name+'conv1',
                 padding='SAME', activation=None)(net)
    net_out_1 = net

    net = MaxPool2D([5, 5], strides=1,  name=name+'pool2', padding='SAME')(net)
    net = Conv2D(n_filters, 3, name=name+'conv2',
                 padding='SAME', activation=None)(net)
    net_out_2 = net

    return Add(name=name+'sum')([net_relu, net_out_1, net_out_2])


def MultiResolutionFusion(high_inputs=None, low_inputs=None, n_filters=256, name=''):
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

    if low_inputs is None:  # RefineNet block 4

        #fuse = Conv2D(n_filters, 3, name=name+'conv', activation=None)(high_inputs)

        #return fuse

        return high_inputs

        # return UpSampling2D(size=2, interpolation='bilinear', name=name+'up')(low_inputs)

    else:

        conv_low = Conv2D(n_filters, 3, padding='same', name=name+'conv_lo', activation=None,
                          kernel_initializer=kern_init, kernel_regularizer=kern_reg)(low_inputs)
        conv_low = BatchNormalization()(conv_low)

        conv_high = Conv2D(n_filters, 3, padding='same', name=name+'conv_hi', activation=None,
                           kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high_inputs)
        conv_high = BatchNormalization()(conv_high)
        # conv_low_up = BilinearUpsampling(conv_low,2)
        # conv_high_up = UpSampling2D(size=2, interpolation='bilinear', name=name+'up')(conv_high)

        # DYNAMIC SCALING HEREE

        # print("low_dim", keras.backend.int_shape(conv_low))
        # print("high_dim", keras.backend.int_shape(conv_high))

        #conv_low_up = UpSampling2D(size=2, interpolation='bilinear', name=name+'up')(conv_low)
        #return Add(name=name+'sum')([conv_low, conv_high])

        low_dim = keras.backend.int_shape(conv_low)[1:3]
        high_dim = keras.backend.int_shape(conv_high)[1:3]
        print("low_dim", low_dim)
        print("high_dim", high_dim)
        mysize = (high_dim[0]//low_dim[0], high_dim[1]//low_dim[1])
        print("SIZE", mysize)
        low_up = UpSampling2D(size=mysize, interpolation='bilinear')(conv_low)
        out = Add()([conv_high, low_up])

        # return Add(name=name+'sum')([conv_low_up, conv_high])

        # sum = ScaledAdd()(conv_high, conv_low)

        return out


def RefineBlock(high_inputs=None, low_inputs=None, block=0):
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

    if low_inputs is None:  # block 4
        rcu_high = ResidualConvUnit(
            high_inputs, n_filters=512, name='rf_{}_rcu_h1_'.format(block))
        rcu_high = ResidualConvUnit(
            rcu_high, n_filters=512, name='rf_{}_rcu_h2_'.format(block))

        fuse = MultiResolutionFusion(
            high_inputs=rcu_high, low_inputs=None, n_filters=512, name='rf_{}_mrf_'.format(block))

        fuse_pooling = ChainedResidualPooling(
            fuse, n_filters=512, name='rf_{}_crp_'.format(block))
        output = ResidualConvUnit(
            fuse_pooling, n_filters=512, name='rf_{}_rcu_o1_'.format(block))
        return output
    else:
        high_n = keras.backend.int_shape(high_inputs)[-1]
        low_n = keras.backend.int_shape(low_inputs)[-1]
        # high_n = 256
        # low_n = 256

        rcu_high = ResidualConvUnit(
            high_inputs, n_filters=high_n, name='rf_{}_rcu_h1_'.format(block))
        rcu_high = ResidualConvUnit(
            rcu_high, n_filters=high_n, name='rf_{}_rcu_h2_'.format(block))

        # we want 3x RCU's between the pooling of one block
        # and the fusion of the next. Therefore we must run
        # the low_inputs through 2x RCU as well

        rcu_low = ResidualConvUnit(
            low_inputs, n_filters=low_n, name='rf_{}_rcu_l1_'.format(block))
        rcu_low = ResidualConvUnit(
            rcu_low, n_filters=low_n, name='rf_{}_rcu_l2_'.format(block))

        fuse = MultiResolutionFusion(
            rcu_high, rcu_low, n_filters=256, name='rf_{}_mrf_'.format(block))

        fuse_pooling = ChainedResidualPooling(
            fuse, n_filters=256, name='rf_{}_crp_'.format(block))
        output = ResidualConvUnit(
            fuse_pooling, n_filters=256, name='rf_{}_rcu_o1_'.format(block))
        return output


def build_refinenet(input_shape, num_classes, is_training=True, frontend_trainable=False, tf_frontend=True, out_logits=True):
    """
    Builds the RefineNet model.

    Arguments:
      inputs: The input tensor
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction
      num_classes: Number of classes

    Returns:
      RefineNet model
    """

    high = [None, None, None, None]
    low = [None, None, None]

    # set the frontend and retrieve high
    frontend = None


    from classification_models.tfkeras import Classifiers

    ResNet18, preprocess_input = Classifiers.get('resnet18')
    frontend = ResNet18((224, 224, 3), weights='imagenet')


    #print(frontend.summary())

    high[0] = frontend.get_layer("add_7").output
    high[1] = frontend.get_layer("add_5").output
    high[2] = frontend.get_layer("add_3").output
    high[3] = frontend.get_layer("add_1").output


    #high[0] = frontend.get_layer("add_15").output
    #high[1] = frontend.get_layer("add_12").output
    #high[2] = frontend.get_layer("add_6").output
    #high[3] = frontend.get_layer("add_2").output

    '''
    if tf_frontend:  # attempt to use the ResNet implementation provided by TensorFlow

        frontend = ResNet50(input_shape=input_shape,
                                include_top=False, weights='imagenet')

        layer_names = [l.name for l in frontend.layers]

        #print(frontend.summary())

        # model.get_layer(layer_name).output

        # get the output of conv block
        for cb in range(2, 6):
            regex = re.compile("conv{}_.+_out".format(cb))
            last_block_layer = max(
                [name for name in layer_names if re.match(regex, name)])
            block_out = frontend.get_layer(last_block_layer).output
            high[5-cb] = (block_out)

        print(high)
    else:  # Use implementation at resnet_101.py from https://github.com/Attila94/refinenet-keras/blob/master/model/resnet_101.py
        resnet_weights = 'models/resnet101_weights_tf.h5'

        frontend = resnet101_model(input_shape, resnet_weights)

        # Get ResNet block output layers
        high = [frontend.get_layer('res5c_relu').output,
            frontend.get_layer('res4b22_relu').output,
            frontend.get_layer('res3b3_relu').output,
            frontend.get_layer('res2c_relu').output]
    '''

    
    # Get the feature maps to the proper size with bottleneck
    high[0] = Conv2D(512, 1, padding='same', name='resnet_map1', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[0])
    high[1] = Conv2D(256, 1, padding='same', name='resnet_map2', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[1])
    high[2] = Conv2D(256, 1, padding='same', name='resnet_map3', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[2])
    high[3] = Conv2D(256, 1, padding='same', name='resnet_map4', kernel_initializer=kern_init, kernel_regularizer=kern_reg)(high[3])
    
    #for h in high:      # this dont do shit bc the author doesnt understand references in python. Whoops.
    #    h = BatchNormalization()(h)
    
#    for h in range(len(high)):
#        high[h] = BatchNormalization()(high[h])


    # RefineNet
    # print("HIGH 0", high[0])
    low[0] = RefineBlock(high_inputs=high[0],low_inputs=None, block=4) # Only input ResNet 1/32
    low[1] = RefineBlock(high_inputs=high[1],low_inputs=low[0], block=3) # High input = ResNet 1/16, Low input = Previous 1/16
    low[2] = RefineBlock(high_inputs=high[2],low_inputs=low[1], block=2) # High input = ResNet 1/8, Low input = Previous 1/8
    net = RefineBlock(high_inputs=high[3],low_inputs=low[2], block=1) # High input = ResNet 1/4, Low input = Previous 1/4


    net = ResidualConvUnit(net, name='rf_rcu_o1_')
    net = ResidualConvUnit(net, name='rf_rcu_o2_')

    net = UpSampling2D(size=4, interpolation='bilinear', name='rf_up_o')(net)

#    net = Conv2D(num_classes, 1, activation = 'softmax', name='rf_pred')(net)
    net = Conv2D(num_classes, 1, activation = None, name='rf_logits')(net)


    model = Model(frontend.input, net)

    

    for layer in model.layers:
        if 'rb' in layer.name or 'rf_' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = frontend_trainable

    print(model.summary())

    return model


'''
# build_refinenet(None, 2)
from pathlib import Path
import os, csv

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
        The number of classes
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values, len(class_names)


dataset_basepath=Path("/media/jetson/Samsung500GB/Semantic-Segmentation-Suite/SpaceNet/")
class_labels, class_colors, num_classes = get_label_info(dataset_basepath / "class_dict.csv")


input_shape=(650,650,3)
random_crop = (224,224,3) #dense prediction tasks recommend multiples of 32 +1
#random_crop = (638, 638, 3)

model = build_refinenet(input_shape, num_classes)

print(model.summary())
'''
