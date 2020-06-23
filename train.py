from helpers import get_label_info
from customGenerator import customGenerator

#from datasetLoader import datasetLoader
from pathlib import Path
from models.RefineNetLite import build_refinenet

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
#from memory_saving_gradients import gradients
#from tensorflow.keras.mixed_precision import experimental as mixed_precision

#from tensorflow_large_model_support import LMS

#from customCallbacks import OutputObserver

import datetime

import tensorflow as tf

from tensorflow import keras as K
import tensorboard
from OutputObserver import OutputObserver

import os

#os.system("capsh --print")

#tf.python.profiler.experimental.server.start(6009)

#tf.compat.v1.disable_eager_execution()
#tf.config.experimental.set_lms_enabled(True)

#tf.compat.v1.disable_v2_behavior()

#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

#tf.debugging.set_log_device_placement(True)


gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

#tf.config.threading.set_inter_op_parallelism_threads(512)
#tf.config.threading.set_intra_op_parallelism_threads(512)

#dataset_basepath=Path(r"\Users\jschaffer\Semantic-Segmentation-Suite\Semantic-Segmentation-Suite\SpaceNet\bak")
dataset_basepath=Path("SpaceNet/")
train_images = dataset_basepath / 'train'
train_masks = dataset_basepath / 'train_labels'
val_images = dataset_basepath / 'val'
val_masks = dataset_basepath / 'val_labels'
class_dict = dataset_basepath / 'class_dict.csv'

class_labels, class_colors, num_classes = get_label_info(dataset_basepath / "class_dict.csv")

print("num classes,", num_classes)

print("labels", class_labels)

print("class_colors",class_colors)


input_shape=(650,650,3)
random_crop = (224,224,3) #dense prediction tasks recommend multiples of 32 +1
#random_crop = (638, 638, 3)
batch_size = 1
epochs = 100
validation_images = 10

#myTrainGen = datasetLoader(batch_size, train_images, train_masks, num_classes, input_shape, dict(), class_colors, random_crop = random_crop)
#myValGen = datasetLoader(batch_size, val_images, val_masks, num_classes, input_shape, dict(), class_colors, random_crop = random_crop)

myTrainGen = customGenerator(batch_size, train_images, train_masks, num_classes, input_shape, dict(), class_colors, random_crop = random_crop)
myValGen = customGenerator(batch_size, val_images, val_masks, num_classes, input_shape, dict(), class_colors, random_crop = random_crop)

#train_ds = tf.data.Dataset.from_generator(myTrainGen.generator, (np.float32, np.float32))
#val_ds = tf.data.Dataset.from_generator(myValGen.generator, (np.float32, np.float32))

#train_ds = myTrainGen.prepare_for_training()
#val_ds = myValGen.prepare_for_training()

steps_per_epoch = myTrainGen.num_samples // batch_size


input_shape = random_crop if random_crop else input_shape # adjust network input for random cropping

model = build_refinenet(input_shape, num_classes)


def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
        Kweights = tf.constant(weights)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return K.losses.categorical_crossentropy(y_true, y_pred, from_logits=True) * K.backend.sum(y_true *Kweights, axis=-1)
    return wcce


weights=[1.0,1.5,0.5]
#model.compile(optimizer = Adam(lr = 1e-4), loss = CategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
model.compile(optimizer = Adam(lr = 1e-4), loss=weighted_categorical_crossentropy([1.0,1.5,0.5]), metrics = ['accuracy'])


#, tf.keras.metrics.MeanIoU(num_classes=2)]


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)



#lms_callback= LMS()

#swapout_threshold=40, swapin_ahead=3, swapin_groupby=2)

#lms_callback.batch_size = 1
#lms_callback.run()

tmp_data = next(myValGen.generator())
save_imgs = OutputObserver(tmp_data, log_dir, class_colors)



model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)
#print("ds_type", type(train_ds))


import matplotlib.pyplot as plt

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


#display(next(myTrainGen.generator()))



model.fit(x=myTrainGen.generator(),
          steps_per_epoch = steps_per_epoch,
          validation_data = myValGen.generator(),
          validation_steps = validation_images // batch_size,
          epochs = epochs,
          #class_weight= {0 : 1.0, 1 : 1.5, 2 : 0.5}, #building is base, perimeter is elevated
          callbacks = [tensorboard_callback, model_checkpoint, save_imgs])

# callbacks = [model_checkpoint, tbCallBack, lrate, history, save_imgs]
