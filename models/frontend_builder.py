import tensorflow as tf
from tensorflow.contrib import slim
import resnet_v2
import os 


def build_frontend(inputs, frontend, is_training=True, pretrained_dir="models"):
    if frontend == 'ResNet50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training, scope='resnet_v2_50')
            frontend_scope='resnet_v2_50'
            # DeepLabV3 requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_50.ckpt'), slim.get_model_variables('resnet_v2_50'))
    elif frontend == 'ResNet101':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_101(inputs, is_training=is_training, scope='resnet_v2_101')
            frontend_scope='resnet_v2_101'
            # DeepLabV3 requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_101.ckpt'), slim.get_model_variables('resnet_v2_101'))
    elif frontend == 'ResNet152':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_152(inputs, is_training=is_training, scope='resnet_v2_152')
            frontend_scope='resnet_v2_152'
            # DeepLabV3 requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_152.ckpt'), slim.get_model_variables('resnet_v2_152'))
    else:
        raise ValueError("Unsupported ResNet model '%s'. This function only supports ResNet 50, ResNet 101, and ResNet 152" % (frontend))

    return logits, end_points, frontend_scope, init_fn 