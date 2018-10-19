import subprocess
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ALL", help='Which model weights to download')
args = parser.parse_args()


if args.model == "ResNet50" or args.model == "ALL":
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/resnet_v2_50.ckpt', "-P", "models"])

if args.model == "ResNet101" or args.model == "ALL":
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/resnet_v2_101.ckpt', "-P", "models"])

if args.model == "ResNet152" or args.model == "ALL":
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/resnet_v2_152.ckpt', "-P", "models"])

if args.model == "MobileNetV2" or args.model == "ALL":
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/mobilenet_v2.ckpt.meta', "-P", "models"])
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/mobilenet_v2.ckpt.index', "-P", "models"])
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/mobilenet_v2.ckpt.data-00000-of-00001', "-P", "models"])

if args.model == "InceptionV4" or args.model == "ALL":
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/inception_v4.ckpt', "-P", "models"])
	
if args.model == "SEResNeXt50" or args.model == "ALL":
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/se_resnext50.ckpt.meta', "-P", "models"])
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/se_resnext50.ckpt.index', "-P", "models"])
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/se_resnext50.ckpt.data-00000-of-00001', "-P", "models"])

if args.model == "SEResNeXt101" or args.model == "ALL":
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/se_resnext101.ckpt.meta', "-P", "models"])
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/se_resnext101.ckpt.index', "-P", "models"])
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/se_resnext101.ckpt.data-00000-of-00001', "-P", "models"])
