import subprocess

subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz'])
subprocess.check_output(['tar', '-xvf', 'resnet_v1_50_2016_08_28.tar.gz'])
subprocess.check_output(['mv', 'resnet_v1_50.ckpt', 'models'])
subprocess.check_output(['rm', 'resnet_v1_50_2016_08_28.tar.gz'])

subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz'])
subprocess.check_output(['tar', '-xvf', 'resnet_v1_101_2016_08_28.tar.gz'])
subprocess.check_output(['mv', 'resnet_v1_101.ckpt', 'models'])
subprocess.check_output(['rm', 'resnet_v1_101_2016_08_28.tar.gz'])

subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'])
subprocess.check_output(['tar', '-xvf', 'resnet_v1_152_2016_08_28.tar.gz'])
subprocess.check_output(['mv', 'resnet_v1_152.ckpt', 'models'])
subprocess.check_output(['rm', 'resnet_v1_152_2016_08_28.tar.gz'])
