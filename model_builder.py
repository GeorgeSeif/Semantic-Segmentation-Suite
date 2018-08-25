import sys, os
import tensorflow as tf

sys.path.append("models")
from FC_DenseNet_Tiramisu import build_fc_densenet
from Encoder_Decoder import build_encoder_decoder
from RefineNet import build_refinenet
from FRRN import build_frrn
from MobileUNet import build_mobile_unet
from PSPNet import build_pspnet
from GCN import build_gcn
from DeepLabV3 import build_deeplabv3
from DeepLabV3_plus import build_deeplabv3_plus
from AdapNet import build_adaptnet

SUPPORTED_MODELS = ["FC-DenseNet56", "FC-DenseNet67", "FC-DenseNet103", "Encoder-Decoder", "Encoder-Decoder-Skip", "RefineNet-Res50", "RefineNet-Res101", "RefineNet-Res152",
    "FRRN-A", "FRRN-B", "MobileUNet", "MobileUNet-Skip", "PSPNet-Res50", 'PSPNet-Res101', "PSPNet-Res152", "GCN-Res50", "GCN-Res101", "GCN-Res152", "DeepLabV3-Res50",
    "DeepLabV3-Res101", "DeepLabV3-Res152", "DeepLabV3_plus-Res50", "DeepLabV3_plus-Res101", "DeepLabV3_plus-Res152", "AdapNet", "custom"]

def download_checkpoints(model_name):
    subprocess.check_output(["python", "get_pretrained_checkpoints.py", "--model=" + model_name])



def build_model(model_name, net_input, num_classes):
	# Get the selected model. 
	# Some of them require pre-trained ResNet

	print("Preparing the model ...")

	if model_name not in SUPPORTED_MODELS:
		raise ValueError("The model you selelect is not supported. The following models are currently supported: {0}".format(SUPPORTED_MODELS))

	if "Res50" in model_name and not os.path.isfile("models/resnet_v2_50.ckpt"):
	    download_checkpoints("Res50")
	if "Res101" in model_name and not os.path.isfile("models/resnet_v2_101.ckpt"):
	    download_checkpoints("Res101")
	if "Res152" in model_name and not os.path.isfile("models/resnet_v2_152.ckpt"):
	    download_checkpoints("Res152")

	network = None
	init_fn = None
	if model_name == "FC-DenseNet56" or model_name == "FC-DenseNet67" or model_name == "FC-DenseNet103":
	    network = build_fc_densenet(net_input, preset_model = model_name, num_classes=num_classes)
	elif model_name == "RefineNet-Res50" or model_name == "RefineNet-Res101" or model_name == "RefineNet-Res152":
	    # RefineNet requires pre-trained ResNet weights
	    network, init_fn = build_refinenet(net_input, preset_model = model_name, num_classes=num_classes)
	elif model_name == "FRRN-A" or model_name == "FRRN-B":
	    network = build_frrn(net_input, preset_model = model_name, num_classes=num_classes)
	elif model_name == "Encoder-Decoder" or model_name == "Encoder-Decoder-Skip":
	    network = build_encoder_decoder(net_input, preset_model = model_name, num_classes=num_classes)
	elif model_name == "MobileUNet" or model_name == "MobileUNet-Skip":
	    network = build_mobile_unet(net_input, preset_model = model_name, num_classes=num_classes)
	elif model_name == "PSPNet-Res50" or model_name == "PSPNet-Res101" or model_name == "PSPNet-Res152":
	    # Image size is required for PSPNet
	    # PSPNet requires pre-trained ResNet weights
	    network, init_fn = build_pspnet(net_input, label_size=[args.crop_height, args.crop_width], preset_model = model_name, num_classes=num_classes)
	elif model_name == "GCN-Res50" or model_name == "GCN-Res101" or model_name == "GCN-Res152":
	    # GCN requires pre-trained ResNet weights
	    network, init_fn = build_gcn(net_input, preset_model = model_name, num_classes=num_classes)
	elif model_name == "DeepLabV3-Res50" or model_name == "DeepLabV3-Res101" or model_name == "DeepLabV3-Res152":
	    # DeepLabV requires pre-trained ResNet weights
	    network, init_fn = build_deeplabv3(net_input, preset_model = model_name, num_classes=num_classes)
	elif model_name == "DeepLabV3_plus-Res50" or model_name == "DeepLabV3_plus-Res101" or model_name == "DeepLabV3_plus-Res152":
	    # DeepLabV3+ requires pre-trained ResNet weights
	    network, init_fn = build_deeplabv3_plus(net_input, preset_model = model_name, num_classes=num_classes)
	elif model_name == "AdapNet":
	    network = build_adaptnet(net_input, num_classes=num_classes)
	elif model_name == "custom":
	    network = build_custom(net_input, num_classes)
	else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

	return network, init_fn