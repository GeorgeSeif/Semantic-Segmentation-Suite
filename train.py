import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorboard
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from Callbacks import OutputObserver
from customGenerator import customGenerator
from models.RefineNetLite import build_refinenet
from utils.helpers import get_label_info
from utils.logging import CsvLogger


# Tensorflow configuration
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


parser = argparse.ArgumentParser()
parser.add_argument('--continue_model', type=str, default=None,
                    help='string of SavedModel folder e.g. \'epoch_22-val_acc_0.8959851264953613\'')
parser.add_argument('--csv', type=str, default=None, help='path to logging csv')
parser.add_argument('--loss', type=str, default='CCE', help='One of \'CCE\' or \'IOU\'')
args = parser.parse_args()


# Data configuration
dataset_basepath = Path("/media/jetson/Samsung500GB/SpaceNet/")
if args.loss == 'CCE':
    dataset_basepath = dataset_basepath / '3class'
else:
    dataset_basepath = dataset_basepath / '2class'

train_images = dataset_basepath / 'train'
train_masks = dataset_basepath / 'train_labels'
val_images = dataset_basepath / 'val'
val_masks = dataset_basepath / 'val_labels'
class_dict = dataset_basepath / 'class_dict.csv'

class_labels, class_colors, num_classes = get_label_info(
    dataset_basepath / "class_dict.csv")

print("num classes,", num_classes)

print("labels", class_labels)

print("class_colors", class_colors)


input_shape = (650, 650, 3)
random_crop = (256, 256, 3)
batch_size = 2
epochs = 200
validation_images = 100


myTrainGen = customGenerator(batch_size, train_images, train_masks,
                             num_classes, input_shape, dict(), class_colors, random_crop=random_crop)
myValGen = customGenerator(batch_size, val_images, val_masks, num_classes,
                           input_shape, dict(), class_colors, random_crop=random_crop)

train_ds = tf.data.Dataset.from_generator(
    myTrainGen.generator, (tf.float32, tf.float32)).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = tf.data.Dataset.from_generator(
    myTrainGen.generator, (tf.float32, tf.float32)).prefetch(tf.data.experimental.AUTOTUNE)

steps_per_epoch = myTrainGen.num_samples // batch_size


# adjust network input for random cropping
input_shape = random_crop if random_crop else input_shape


# load model
start_epoch = 0
model = None
if args.continue_model is not None:
    model = tf.keras.models.load_model(
        "saved_models/{}".format(args.continue_model))  # , compile=False)
    start_epoch = int(args.continue_model[6:8]) + 1
else:
    model = build_refinenet(input_shape, num_classes)

def loss_iou(y, logits):
    ''' 
    differentiable approximation of intersection over union by Y.Wang et al
    works only for binary segmentation
    '''
    #inputs have dimension [batch x height x width x 2] for building/background
    y = tf.reshape(y[:,:,:,0], [-1])
    logits = tf.reshape(logits[:,:,:,0], [-1])

    inter=tf.math.reduce_sum(tf.math.multiply(logits, y))

    union = tf.math.reduce_sum(tf.math.subtract(tf.math.add(logits, y), tf.math.multiply(logits,y)))

    loss = tf.math.subtract(tf.constant(1.0, dtype=tf.float32),tf.math.divide(inter,union))

    return loss

def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
        Kweights = tf.constant(weights)
        if not tf.is_tensor(y_pred):
            y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return K.losses.categorical_crossentropy(y_true, y_pred, from_logits=True) * K.backend.sum(y_true * Kweights, axis=-1)
    return wcce


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

csv_logger = CsvLogger(args.csv)

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = None
if args.loss == 'CCE':
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
else:
    loss_fn = loss_iou



# Prepare the metrics.
train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy('train_acc')

val_loss_metric = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
val_acc_metric = tf.keras.metrics.CategoricalAccuracy('val_acc')
val_iou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)


tmp_data = next(myValGen.generator())
save_imgs = OutputObserver(tmp_data, log_dir, class_colors)
save_imgs.model = model


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_loss_metric(loss_value)
    train_acc_metric.update_state(y, logits)
    return train_loss_metric.result()


@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)  # model.predict(x) #
    loss_value = loss_fn(y, val_logits)

    val_loss_metric(loss_value)
    val_acc_metric.update_state(y, val_logits)

    print("y shape", tf.keras.backend.int_shape(y))
    print("logits shape", tf.keras.backend.int_shape(val_logits))
    print("argmax shape", tf.keras.backend.int_shape(val_logits))
    val_iou_metric.update_state(
        y, tf.keras.backend.softmax(val_logits, axis=-1))



start = datetime.utcnow()

writer = tf.summary.create_file_writer(log_dir)


#start_epoch = 180

for epoch in range(start_epoch, epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_ds):  # (myTrainGen.generator()

        loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 100 batches.
        if step % 100 == 0:

            #save_imgs.on_batch_end(step, epoch)

            print(
                "Step %d / %d: train_loss=%.4f"
                % (step, steps_per_epoch, float(loss_value))
            )
            temp = datetime.utcnow()
            print("frames/second: {}".format(200/(temp-start).total_seconds()))
            start = temp

        if step == steps_per_epoch:
            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            train_loss = train_loss_metric.result()
            print("Epoch %d: train_acc=%.4f train_loss=%.4f" %
                  (epoch, float(train_acc), float(train_loss)))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            train_loss_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in val_ds.take(validation_images):
                test_step(x_batch_val, y_batch_val)

            val_acc = val_acc_metric.result()
            val_loss = val_loss_metric.result()
            val_iou = val_iou_metric.result()

            val_acc_metric.reset_states()
            val_loss_metric.reset_states()
            val_iou_metric.reset_states()
            print("val_acc=%.4f val_loss=%.4f val_iou=%.4f" %
                  (float(val_acc), float(val_loss), float(val_iou)))

            # with writer.as_default():
            #    tf.summary.scalar("val_acc_epoch", val_acc, step=epoch)
            #    tf.summary.scalar("train_acc_epoch", train_acc, step=epoch)

            csv_logger.writeEpoch(epoch, float(train_loss), float(train_acc), float(val_loss), float(val_acc), float(val_iou))

            print("Saving model...")
            save_start = datetime.utcnow()
            model.save('saved_models/epoch_%d-acc_%.4f-loss_%.4f-iou_%.4f' % (epoch, float(val_acc), float(val_loss), float(val_iou)))
            print("Model saved in {} seconds".format((datetime.utcnow()-save_start).total_seconds()))
            
            break