# Semantic Segmentation Suite in TensorFlow

## News

- The current version of this repo works with grayscale labels. I.e the labels only have one channel, where each pixel has the value of its class (so for 12 classes, the labels are integers 0 to 11). I'm currently working on extending this to work with colour image labels.

- The DeepLabV3 model is on the way!

## Description
This repository serves as a Semantic Segmentation Suite. The goal is to easily be able to implement, train, and test new Semantic Segmentation models! Complete with the following:

- Training and testing modes
- Data augmentation
- Several state-of-the-art models. Easily **plug and play** with different models
- Able to use **any** dataset
- Evaluation including precision, recall, f1 score, average accuracy, per-class accuracy, and mean IoU
- Plotting of loss function and accuracy over epochs

**Any suggestions to improve this repository, including any new segmentation models you would like to see are welcome!**

## Models
The following models are currently made available:

- [Encoder-Decoder based on SegNet](https://arxiv.org/abs/1511.00561). This network uses a VGG-style encoder-decoder, where the upsampling in the decoder is done using transposed convolutions.

- [Encoder-Decoder with skip connections based on SegNet](https://arxiv.org/abs/1511.00561). This network uses a VGG-style encoder-decoder, where the upsampling in the decoder is done using transposed convolutions. In addition, it employs additive skip connections from the encoder to the decoder. 

- [Mobile UNet for Semantic Segmentation](https://arxiv.org/abs/1704.04861). Combining the ideas of MobileNets Depthwise Separable Convolutions with UNet to build a high speed, low parameter Semantic Segmentation model.

- [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105). In this paper, the capability of global context information by different-region based context aggregation is applied through a pyramid pooling module together with the proposed pyramid scene parsing network (PSPNet).

- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326). Uses a downsampling-upsampling style encoder-decoder network. Each stage i.e between the pooling layers uses dense blocks. In addition, it concatenated skip connections from the encoder to the decoder. In the code, this is the FC-DenseNet model.

- [RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612). A multi-path refinement network that explicitly exploits all the information available along the down-sampling process to enable high-resolution prediction using long-range residual connections. In this way, the deeper layers that capture high-level semantic features can be directly refined using fine-grained features from earlier convolutions.

- [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323). Combines multi-scale context with pixel-level accuracy by using two processing streams within the network. The residual stream carries information at the full image resolution, enabling precise adherence to segment boundaries. The pooling stream undergoes a sequence of pooling operations
to obtain robust features for recognition. The two streams are coupled at the full image resolution using residuals. In the code, this is the FRRN model.

- Or make your own and plug and play!

**Note:** If you are using any of the networks that rely on a pre-trained ResNet, then you will need to download the pre-trained weights using the provided script. These are currently: PSPNet, RefineNet.


## Files and Directories


- **main.py:** Training and Testing

- **helper.py:** Quick helper functions for data preparation and visualization

- **utils.py:** Utilities for printing, debugging, testing, and evaluation

- **get_pretrained_checkpoints.py:** Downloads the pre-trained ResNet weights for ResNet50, ResNet101, and ResNet152

- **models:** Folder containing all model files. Use this to build your models, or use a pre-built one

- **CamVid:** The CamVid datatset for Semantic Segmentation as a test bed. This is the 11 class version

- **checkpoints:** Checkpoint files for each epoch during training

- **Test:** Test results including images, per-class accuracies, precision, recall, and f1 score


## Installation
This project has the following dependencies:

- Numpy `sudo pip install numpy`

- OpenCV Python `sudo apt-get install python-opencv`

- TensorFlow `sudo pip install --upgrade tensorflow-gpu`

## Usage
The only thing you have to do to get started is set up the folders in the following structure:

    ├── "dataset_name"                   
    |   ├── train
    |   ├── train_labels
    |   ├── val
    |   ├── val_labels
    |   ├── test
    |   ├── test_labels

Put a text file under the dataset directory called "class_list" which contains the list of classes, one on each line like so:

```
Sky
Building
Pole
```

Then you can simply run `main.py`! Check out the optional command line arguments:

```
optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --is_training IS_TRAINING
                        Whether we are training or testing
  --continue_training CONTINUE_TRAINING
                        Whether to continue training from a checkpoint
  --dataset DATASET     Dataset you are using.
  --crop_height CROP_HEIGHT
                        Height of cropped input image to network
  --crop_width CROP_WIDTH
                        Width of cropped input image to network
  --batch_size BATCH_SIZE
                        Number of images in each batch
  --num_val_images NUM_VAL_IMAGES
                        The number of images to used for validations
  --h_flip H_FLIP       Whether to randomly flip the image horizontally for
                        data augmentation
  --v_flip V_FLIP       Whether to randomly flip the image vertically for data
                        augmentation
  --brightness BRIGHTNESS
                        Whether to randomly change the image brightness for
                        data augmentation
  --model MODEL         The model you are using. Currently supports: FC-
                        DenseNet56, FC-DenseNet67, FC-DenseNet103, Encoder-
                        Decoder, Encoder-Decoder-Skip, RefineNet-Res101,
                        RefineNet-Res152, FRRN-A, FRRN-B, MobileUNet,
                        MobileUNet-Skip, PSPNet, custom


```
    

## Results

These are some **sample results**.

In training, I used a batch size of 1 and image size of 352x480. The following results are for the FC-DenseNet103 model trained for 300 epochs. I used RMSProp with learning rate 0.001 and decay 0.995. I **did not** use any data augmentation like in the paper.

**Note that the checkpoint files are not uploaded to this repository since they are too big for GitHub (greater than 100 MB)**


| Class 	| Original Accuracy  	| My Accuracy |
| ------------- 		| ------------- | -------------|
| Sky  		| 93.0 | 94.1  |
| Building 		| 83.0  | 81.2  |
| Pole  		| 37.8  | 38.3  |
| Road 		| 94.5  | 97.5  |
| Pavement  		| 82.2  | 87.9  |
| Tree 		| 77.3  | 75.5  |
| SignSymbol  		| 43.9  | 49.7  |
| Fence 		| 37.1  | 69.0  |
| Car  		| 77.3  | 87.0  |
| Pedestrian 		| 59.6  | 60.3  |
| Bicyclist  		| 50.5  | 75.3  |
| Unlabelled 		| N/A  | 40.9  |
| Global  		| 91.5 | 86.0  |


Loss vs Epochs            |  Val. Acc. vs Epochs
:-------------------------:|:-------------------------:
![alt text-1](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/loss_vs_epochs.png)  |  ![alt text-2](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/accuracy_vs_epochs.png)


Original            |  GT   |  Result
:-------------------------:|:-------------------------:|:-------------------------:
![alt-text-3](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/0001TP_008550.png "Original")  |  ![alt-text-4](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/0001TP_008550_gt.png "GT")  |   ![alt-text-5](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/0001TP_008550_pred.png "Result")

