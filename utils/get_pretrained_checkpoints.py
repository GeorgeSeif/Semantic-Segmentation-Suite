import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ALL", help='Which model weights to download')
args = parser.parse_args()


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
	subprocess.check_output(
		['wget', 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz', "-P", "models"])
	try:
		subprocess.check_output(['tar', '-xvf', 'models/inception_v4_2016_09_09.tar.gz', "-C", "models"])
		subprocess.check_output(['rm', 'models/inception_v4_2016_09_09.tar.gz'])
	except Exception as e:
		print(e)
		pass
