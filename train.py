from helpers import get_label_info
from customGenerator import customGenerator
from pathlib import Path
from models.RefineNet import build_refinenet

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


import tensorflow as tf




dataset_basepath=Path("SpaceNet/")
train_images = dataset_basepath / 'train'
train_masks = dataset_basepath / 'train_labels'
val_images = dataset_basepath / 'val'
val_masks = dataset_basepath / 'val_labels'
class_dict = dataset_basepath / 'class_dict.csv'

class_labels, class_colors, num_classes = get_label_info(dataset_basepath / "class_dict.csv")


input_shape=(650,650,3)
random_crop = (448,448,3) #dense prediction tasks recommend multiples of 32 +1
#random_crop = (638, 638, 3)


batch_size = 1
epochs = 5
validation_images = 4

myTrainGen = customGenerator(batch_size, train_images, train_masks, num_classes, input_shape, dict(), class_colors, random_crop = random_crop)
myValGen = customGenerator(batch_size, val_images, val_masks, num_classes, input_shape, dict(), class_colors, random_crop = random_crop)


steps_per_epoch = myTrainGen.num_samples // batch_size


input_shape = random_crop if random_crop else input_shape # adjust network input for random cropping

model = build_refinenet(input_shape, num_classes)
model.compile(optimizer = Adam(lr = 1e-4), loss = CategoricalCrossentropy(), metrics = ['accuracy'])

model.fit(x=myTrainGen.generator(), 
          batch_size = batch_size,
          steps_per_epoch = steps_per_epoch,
          validation_data = myValGen.generator(),
          validation_steps = validation_images // batch_size,
          epochs = epochs)

# callbacks = [model_checkpoint, tbCallBack, lrate, history, save_imgs]