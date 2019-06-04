from __future__ import print_function
import os, time, sys, math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
import cv2
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg') # to generate plot without X server

from utils.utils import data_augmentation, crop_image_and_label, str2bool
from utils import utils, helpers
from builders import model_builder
from utils.parser import parse

args = parse()

#######################################################################################
# prepare model and dataset
#######################################################################################

class_names_list, label_values_list, class_names_string = helpers.get_label_info(os.path.join(args.dataset, "class_dict_colorfull.csv"))
nb_class = len(class_names_list)

# retrieve dataset file names
dataset_file_name = utils.get_dataset_file_name( dataset_dir=args.dataset )

input_size = (
    utils.get_minimal_size( dataset_dir=args.dataset )
    if (not args.crop_height and not args.crop_width)
    else {'height':args.crop_height, 'width':args.crop_width} )

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Compute your softmax cross entropy loss
net_input = tf.placeholder(tf.float32,shape=[args.batch_size, input_size['height'], input_size['width'], 3])
net_output = tf.placeholder(tf.float32,shape=[args.batch_size, input_size['height'], input_size['width'], nb_class])

# load the model
network, init_fn = model_builder.build_model(
    model_name=args.model,
    frontend=args.frontend,
    net_input=net_input,
    num_classes=nb_class,
    image_width=input_size['width'],
    image_height=input_size['height'],
    is_training=True)

loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output) )

optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate, decay=args.regularization).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver = tf.train.Saver(max_to_keep=1000)
sess.run( tf.global_variables_initializer() )

model_parameters = utils.count_params()
print("This model has %d trainable parameters"% (model_parameters))

# If pre-trained ResNet required, load weights (must be done AFTER variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)


print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Input Size --> %s x %s = %s" %(input_size['width'], input_size['height'], input_size['width']*input_size['height']) )
print("Num Epochs -->", args.nb_epoch)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", nb_class)
print("Learning rate -->", args.learning_rate)
print("Regularization -->", args.regularization)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")


#######################################################################################
# train the model
#######################################################################################

avg_loss_per_epoch, avg_scores_per_epoch, avg_iou_per_epoch = [], [], []

# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(dataset_file_name['validation']['input']))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices=random.sample(range(0,len(dataset_file_name['validation']['input'])),num_vals)

# Do the training here
for epoch in range(args.epoch_start_i, args.nb_epoch):

    current_losses, cnt = [], 0

    id_list = np.random.permutation( len(dataset_file_name['training']['input']) ) # Equivalent to shuffling

    nb_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st=time.time()

    for i in range(nb_iters):

        input_image_batch, output_image_batch = [], []
        
        for j in range(args.batch_size): # Collect a batch of images

            index = i*args.batch_size + j
            id = id_list[index]
            input_image = utils.load_image( dataset_file_name['training']['input'][id] )
            output_image = utils.load_image( dataset_file_name['training']['output'][id] )

            with tf.device('/cpu:0'):

                #input_image, output_image = data_augmentation( args, input_image, output_image )

                if input_size != None:
                    input_image, output_image = crop_image_and_label( input_image, output_image, input_size )

                input_image = np.float32( input_image ) / 255.0
                output_image = np.float32( helpers.one_hot_it(label=output_image, label_values=label_values_list) )

                input_image_batch.append( np.expand_dims(input_image, axis=0) )
                output_image_batch.append( np.expand_dims(output_image, axis=0) )

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

        # Do the training
        _, current=sess.run([optimizer,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})

        current_losses.append(current)

        cnt = cnt + args.batch_size

        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)

    # Create directories if needed
    if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
        os.makedirs("%s/%04d"%("checkpoints",epoch))

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoch))


    if epoch % args.validation_step == 0:
        print("Performing validation")
        target=open("%s/%04d/val_scores.csv"%("checkpoints",epoch),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))

        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []

        # Do the validation on a small set of validation images
        for ind in val_indices:

            input_image = np.expand_dims(np.float32(utils.load_image(dataset_file_name['validation']['input'][ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0

            gt = utils.load_image(dataset_file_name['validation']['output'][ind])[:args.crop_height, :args.crop_width]

            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values_list))

            output_image = sess.run(network,feed_dict={net_input:input_image})

            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values_list)

            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, nb_class=nb_class)

            file_name = utils.filepath_to_name(dataset_file_name['validation']['input'][ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)

            gt = helpers.colour_code_segmentation(gt, label_values_list)

            file_name = os.path.basename(dataset_file_name['validation']['input'][ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)

        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)
    scores_list = []


    fig1, ax1 = plt.subplots(figsize=(11, 8))

    ax1.plot(range(epoch+1), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")


    plt.savefig('accuracy_vs_epochs.png')

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))

    ax2.plot(range(epoch+1), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")

    plt.savefig('loss_vs_epochs.png')

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoch+1), avg_iou_per_epoch)
    ax3.set_title("Average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")

    plt.savefig('iou_vs_epochs.png')



