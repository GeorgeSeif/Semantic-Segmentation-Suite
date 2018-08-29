import subprocess
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ALL", help='Which model weights to download')
args = parser.parse_args()

###############################
# VGG Net
###############################
# subprocess.check_output(['wget','http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'vgg_16_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'vgg_16.ckpt', 'models'])
# subprocess.check_output(['rm', 'vgg_16_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'vgg_19_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'vgg_19.ckpt', 'models'])
# subprocess.check_output(['rm', 'vgg_19_2016_08_28.tar.gz'])

###############################
# Inception Net
###############################
# subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'inception_v1_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'inception_v1.ckpt', 'models'])
# subprocess.check_output(['rm', 'inception_v1_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'inception_v2_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'inception_v2.ckpt', 'models'])
# subprocess.check_output(['rm', 'inception_v2_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'inception_v3_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'inception_v3.ckpt', 'models'])
# subprocess.check_output(['rm', 'inception_v3_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'inception_v4_2016_09_09.tar.gz'])
# subprocess.check_output(['mv', 'inception_v4.ckpt', 'models'])
# subprocess.check_output(['rm', 'inception_v4_2016_09_09.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'inception_resnet_v2_2016_08_30.tar.gz'])
# subprocess.check_output(['mv', 'inception_resnet_v2.ckpt', 'models'])
# subprocess.check_output(['rm', 'inception_resnet_v2_2016_08_30.tar.gz'])


###############################
# ResNet V1
###############################
# subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'resnet_v1_50_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'resnet_v1_50.ckpt', 'models'])
# subprocess.check_output(['rm', 'resnet_v1_50_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'resnet_v1_101_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'resnet_v1_101.ckpt', 'models'])
# subprocess.check_output(['rm', 'resnet_v1_101_2016_08_28.tar.gz'])

# subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'])
# subprocess.check_output(['tar', '-xvf', 'resnet_v1_152_2016_08_28.tar.gz'])
# subprocess.check_output(['mv', 'resnet_v1_152.ckpt', 'models'])
# subprocess.check_output(['rm', 'resnet_v1_152_2016_08_28.tar.gz'])


###############################
# ResNet V2
###############################
if args.model == "ResNet50" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz', "-P", "models"])
	try:
		subprocess.check_output(['tar', '-xvf', 'models/resnet_v2_50_2017_04_14.tar.gz', "-C", "models"])
		subprocess.check_output(['rm', 'models/resnet_v2_50_2017_04_14.tar.gz'])
	except Exception as e:
		print(e)
		pass

if args.model == "ResNet101" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz', "-P", "models"])
	try:
		subprocess.check_output(['tar', '-xvf', 'models/resnet_v2_101_2017_04_14.tar.gz', "-C", "models"])
		subprocess.check_output(['rm', 'models/resnet_v2_101_2017_04_14.tar.gz'])
	except Exception as e:
		print(e)
		pass

if args.model == "ResNet152" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz', "-P", "models"])
	try:
		subprocess.check_output(['tar', '-xvf', 'models/resnet_v2_152_2017_04_14.tar.gz', "-C", "models"])
		subprocess.check_output(['rm', 'models/resnet_v2_152_2017_04_14.tar.gz'])
	except Exception as e:
		print(e)
		pass

if args.model == "MobileNetV2" or args.model == "ALL":
	subprocess.check_output(['wget','https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz', "-P", "models"])
	try:
		subprocess.check_output(['tar', '-xvf', 'models/mobilenet_v2_1.4_224.tgz', "-C", "models"])
		subprocess.check_output(['rm', 'models/mobilenet_v2_1.4_224.tgz'])
	except Exception as e:
		print(e)
		pass

if args.model == "InceptionV4" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz', "-P", "models"])
	try:
		subprocess.check_output(['tar', '-xvf', 'inception_v4_2016_09_09.tar.gz', "-C", "models"])
		subprocess.check_output(['rm', 'inception_v4_2016_09_09.tar.gz'])
	except Exception as e:
		print(e)
		pass

if args.model == "NASNet" or args.model == "ALL":
	subprocess.check_output(['wget','https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz', "-P", "models"])
	try:
		subprocess.check_output(['tar', '-xvf', 'nasnet-a_large_04_10_2017.tar.gz', "-C", "models"])
		subprocess.check_output(['rm', 'nasnet-a_large_04_10_2017.tar.gz'])
	except Exception as e:
		print(e)
		pass

try:
	subprocess.check_output(['rm', 'train.graph'])
	subprocess.check_output(['rm', 'eval.graph'])
except Exception as e:
		print(e)
		pass