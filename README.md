# FC-DenseNet Tiramisu in TensorFlow

## Description
This is an implementation of the Fully-Convolutional DenseNet for Semantic Segmentation from the paper:

[The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)


## Files and Directories

- **FC_DenseNet_Tiramisu.py:** The model file. Use this to build your custom FC-DenseNet models, or one from the original paper

- **main.py:** Training and Testing on the CamVid dataset

- **helper.py:** Quick helper functions for data preparation and visualization

- **utils.py:** Utilities for printing, debugging, and testing

- **CamVid:** The CamVid datatset for Semantic Segmentation

- **checkpoints:** Checkpoint files for each epoch during training

- **Test:** Test results including images and per-class accuracies


## Installation
This project has the following dependencies:

- Numpy `sudo pip install numpy`

- OpenCV Python `sudo apt-get install python-opencv`

- TensorFlow `sudo pip install --upgrade tensorflow-gpu`


## Results

In training, I used a batch size of 1 and image size of 352x480. The following results are for the FC-DenseNet103 model trained for 250 epochs. I used RMSProp with learning rate 0.001 and decay 0.995. **Note that the checkpoint files are not uploaded to this repository since they are too big for GitHub (greater than 100 MB)


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
| Unlabelled 		| Content Cell  | 40.9  |
| Global  		| 91.5 | 86.0  |


![alt text-1](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/loss_vs_epochs.png) ![alt text-2](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/accuracy_vs_epochs.png)

![alt-text-3](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/0001TP_008550.png "Original") ![alt-text-4](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/0001TP_008550_gt.png "GT") ![alt-text-5](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/0001TP_008550_pred.png "Result")



## TO DO

- Use class weights

- Measure Intersection over Union (IoU)