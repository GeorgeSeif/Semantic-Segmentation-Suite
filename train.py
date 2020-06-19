from helpers import get_label_info
from customGenerator import customGenerator

from datasetLoader import datasetLoader
from pathlib import Path
from models.RefineNet import build_refinenet

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


import tensorflow as tf

from tensorflow import keras
import tensorboard


tf.compat.v1.disable_eager_execution()

dataset_basepath=Path("/Users/jschaffer/Semantic-Segmentation-Suite/Semantic-Segmentation-Suite/SpaceNet/")
train_images = dataset_basepath / 'train'
train_masks = dataset_basepath / 'train_labels'
val_images = dataset_basepath / 'val'
val_masks = dataset_basepath / 'val_labels'
class_dict = dataset_basepath / 'class_dict.csv'

class_labels, class_colors, num_classes = get_label_info(dataset_basepath / "class_dict.csv")


input_shape=(650,650,3)
random_crop = (224,224,3) #dense prediction tasks recommend multiples of 32 +1
#random_crop = (638, 638, 3)


batch_size = 1
epochs = 5
validation_images = 4

myTrainGen = datasetLoader(batch_size, train_images, train_masks, num_classes, input_shape, dict(), class_colors, random_crop = random_crop)
myValGen = datasetLoader(batch_size, val_images, val_masks, num_classes, input_shape, dict(), class_colors, random_crop = random_crop)


train_ds = myTrainGen.prepare_for_training()
val_ds = myValGen.prepare_for_training()

steps_per_epoch = myTrainGen.num_samples // batch_size


input_shape = random_crop if random_crop else input_shape # adjust network input for random cropping

model = build_refinenet(input_shape, num_classes)
model.compile(optimizer = Adam(lr = 1e-4), loss = CategoricalCrossentropy(), metrics = ['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/fit/', profile_batch=3) #update_freq='batch' || 'epoch'


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)
print("ds_type", type(train_ds))



model.fit(train_ds,
          steps_per_epoch = steps_per_epoch,
          validation_data = val_ds,
          validation_steps = validation_images // batch_size,
          epochs = epochs,
          callbacks = [tensorboard_callback, model_checkpoint])

# callbacks = [model_checkpoint, tbCallBack, lrate, history, save_imgs]
