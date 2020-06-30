from tensorflow import keras
from helpers import get_label_info
from customGenerator import customGenerator
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.python.keras import backend as K


gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


dataset_basepath=Path("/media/jetson/Samsung500GB/SpaceNet/")
train_images = dataset_basepath / 'train'
train_masks = dataset_basepath / 'train_labels'
val_images = dataset_basepath / 'val'
val_masks = dataset_basepath / 'val_labels'
class_dict = dataset_basepath / 'class_dict.csv'

input_shape = (650, 650, 3)
# dense prediction tasks recommend multiples of 32 +1
random_crop = (256, 256, 3)
#random_crop = (638, 638, 3)
batch_size = 1
epochs = 100
validation_images = 10

class_labels, class_colors, num_classes = get_label_info(
    dataset_basepath / "class_dict.csv")

myValGen = customGenerator(batch_size, val_images, val_masks, num_classes,
                           input_shape, dict(), class_colors, random_crop=random_crop)





def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
        Kweights = tf.constant(weights)
        if not tf.is_tensor(y_pred):
            y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred, from_logits=True) * K.sum(y_true * Kweights, axis=-1)
    return wcce


model = keras.models.load_model(
    "saved_models/epoch_179-acc_0.9154-loss_0.2435-iou_0.0000", compile=False)
    #"saved_models/epoch_99-acc_0.8825-loss_0.3175-iou_0.0000", compile=False)
    #"saved_models/epoch_87-acc_0.9145-loss_0.2340-iou_0.000", compile=False)
    #"saved_models/epoch_55-val_acc_0.9068875908851624", compile=False)
    #"saved_models/epoch_22-val_acc_0.8959851264953613", compile=False)
    #"checkpoints/weights.74-2719.15.hdf5", compile=False)

weights = [1.0, 1.5, 0.5]
#model.compile(optimizer = Adam(lr = 1e-4), loss = CategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
model.compile(optimizer=Adam(
    lr=1e-4), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#lms_callback= LMS(swapout_threshold=40, swapin_ahead=3, swapin_groupby=2)
#lms_callback.batch_size = batch_size
#lms_callback.run()




def display(display_list, epoch, img_num):
    figure = plt.figure(figsize=(15, 4))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    #plt.show()
    save_str = "test" + str(epoch)+ "_" +str(img_num)
    print("Saving ", save_str)
    plt.savefig(save_str)

    # return figure


def labelVisualize(y_pred, mask_colors):
    """
    Convert prediction to color-coded image.
    """
    x = np.argmax(y_pred, axis=-1)
    colour_codes = np.array(mask_colors)
    img = colour_codes[x.astype('uint8')]
    return img


for i in range(3,20):
    img, truth = next(myValGen.generator())
    y_pred = model.predict(img)


    print("pred_shape",  tf.keras.backend.int_shape(y_pred))
    print("truth_shape",  tf.keras.backend.int_shape(truth))

    pred_img = labelVisualize(y_pred[0], class_colors)
    target_labelled = labelVisualize(truth[0], class_colors)

    print("THIS IS MY I", i)

    display([img[0], target_labelled, pred_img], 179, i)