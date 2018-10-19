import tensorflow as tf
from tensorflow.contrib import slim
from frontends import resnet_v2
from frontends import mobilenet_v2
from frontends import inception_v4
import os 


def build_frontend(inputs, frontend, is_training=True, pretrained_dir="models"):
    if frontend == 'ResNet50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training, scope='resnet_v2_50')
            frontend_scope='resnet_v2_50'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'resnet_v2_50.ckpt'), var_list=slim.get_model_variables('resnet_v2_50'), ignore_missing_vars=True)
    elif frontend == 'ResNet101':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_101(inputs, is_training=is_training, scope='resnet_v2_101')
            frontend_scope='resnet_v2_101'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'resnet_v2_101.ckpt'), var_list=slim.get_model_variables('resnet_v2_101'), ignore_missing_vars=True)
    elif frontend == 'ResNet152':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_152(inputs, is_training=is_training, scope='resnet_v2_152')
            frontend_scope='resnet_v2_152'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'resnet_v2_152.ckpt'), var_list=slim.get_model_variables('resnet_v2_152'), ignore_missing_vars=True)
    elif frontend == 'MobileNetV2':
        with slim.arg_scope(mobilenet_v2.training_scope()):
            logits, end_points = mobilenet_v2.mobilenet(inputs, is_training=is_training, scope='mobilenet_v2', base_only=True)
            frontend_scope='mobilenet_v2'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'mobilenet_v2.ckpt'), var_list=slim.get_model_variables('mobilenet_v2'), ignore_missing_vars=True)
    elif frontend == 'InceptionV4':
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, end_points = inception_v4.inception_v4(inputs, is_training=is_training, scope='inception_v4')
            frontend_scope='inception_v4'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'inception_v4.ckpt'), var_list=slim.get_model_variables('inception_v4'), ignore_missing_vars=True)
    else:
        raise ValueError("Unsupported fronetnd model '%s'. This function only supports ResNet50, ResNet101, ResNet152, and MobileNetV2" % (frontend))

    return logits, end_points, frontend_scope, init_fn 