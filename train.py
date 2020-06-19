from helpers import get_label_info
from customGenerator import customGenerator

from datasetLoader import datasetLoader
from pathlib import Path
from models.RefineNetLite import build_refinenet

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
#from memory_saving_gradients import gradients
#from tensorflow.keras.mixed_precision import experimental as mixed_precision

from tensorflow_large_model_support import LMS

from customCallbacks import OutputObserver

import datetime

import tensorflow as tf

from tensorflow import keras
import tensorboard

#tf.python.profiler.experimental.server.start(6009)

#tf.compat.v1.disable_eager_execution()
#tf.config.experimental.set_lms_enabled(True)

#tf.compat.v1.disable_v2_behavior()

#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

#tf.debugging.set_log_device_placement(True)


#gpus=tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu,True)

#tf.config.threading.set_inter_op_parallelism_threads(512)
#tf.config.threading.set_intra_op_parallelism_threads(512)

dataset_basepath=Path("/media/jetson/Samsung500GB/Semantic-Segmentation-Suite/SpaceNet/")
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

myTrainGen = datasetLoader(batch_size, train_images, train_masks, num_classes, input_shape, dict(), class_colors, random_crop = random_crop)
myValGen = datasetLoader(batch_size, val_images, val_masks, num_classes, input_shape, dict(), class_colors, random_crop = random_crop)

#train_ds = tf.data.Dataset.from_generator(myTrainGen.generator, (np.float32, np.float32))
#val_ds = tf.data.Dataset.from_generator(myValGen.generator, (np.float32, np.float32))

train_ds = myTrainGen.prepare_for_training()
val_ds = myValGen.prepare_for_training()

steps_per_epoch = myTrainGen.num_samples // batch_size


input_shape = random_crop if random_crop else input_shape # adjust network input for random cropping

model = build_refinenet(input_shape, num_classes)


model.compile(optimizer = Adam(lr = 1e-4), loss = CategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

#, tf.keras.metrics.MeanIoU(num_classes=2)]


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=3, update_freq='batch')

#lms_callback= LMS()

#swapout_threshold=40, swapin_ahead=3, swapin_groupby=2)

#lms_callback.batch_size = 1
#lms_callback.run()


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
