import argparse
from utils.utils import str2bool

def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_epoch', type=int, default=300, help='Number of epochs to train for')

    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')

    parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')

    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')

    parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')

    parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')

    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')

    parser.add_argument('--crop_height', type=int, default=0, help='Height of cropped input image to network, default automatically crop to smallest dimension.')

    parser.add_argument('--crop_width', type=int, default=0, help='Width of cropped input image to network, default automatically crop to smallest dimension.')

    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')

    parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')

    parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')

    parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')

    parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')

    parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')

    parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. See model_builder.py for supported models')

    parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='The learning rate')
    
    parser.add_argument('--regularization', type=float, default=1.0, help='The regularization parameter')

    args = parser.parse_args()

    return args