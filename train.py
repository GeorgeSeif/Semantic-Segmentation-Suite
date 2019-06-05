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
#matplotlib.use('Agg') # to generate plot without X server

from utils import data_tool, model_tool, general_tool
from utils.parser import parse
from builders import model_builder

args = parse()


#######################################################################################
# prepare model and dataset
#######################################################################################

class_names_list, label_values_list, class_names_str = data_tool.get_label_info(
    dataset_name=args.dataset,
    dataset_path=args.dataset_path,
    class_file_name="class_dict_colorfull.csv")
nb_class = len(class_names_list)

dataset_file_name = data_tool.get_dataset_file_name( dataset_dir=args.dataset )

input_size = (
    data_tool.get_minimal_size( dataset_dir=args.dataset )
    if (not args.crop_height and not args.crop_width)
    else {'height':args.crop_height, 'width':args.crop_width} )

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Compute your softmax cross entropy loss
net_input = tf.placeholder( tf.float32,shape=[args.batch_size, input_size['height'], input_size['width'], 3] )
net_output = tf.placeholder( tf.float32,shape=[args.batch_size, input_size['height'], input_size['width'], nb_class] )

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

# If pre-trained ResNet required, load weights (must be done AFTER variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

general_tool.display_info( args, input_size, nb_class )


#######################################################################################
# train the model
#######################################################################################

print("\n***** Begin training *****")

avg_loss_per_epoch, avg_scores_per_epoch, avg_iou_per_epoch = [], [], []

# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(dataset_file_name['validation']['input']))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices=random.sample(range(0,len(dataset_file_name['validation']['input'])),num_vals)

for epoch in range(args.epoch_start_i, args.nb_epoch):

    current_losses, cnt = [], 0

    id_list = np.random.permutation( len(dataset_file_name['training']['input']) ) # Equivalent to shuffling

    nb_iters = (
        int( np.floor( len(id_list) / args.batch_size) )
        if args.redux == 1.0
        else int( np.floor( len(id_list) * args.redux / args.batch_size ) ) )

    st = time.time()
    epoch_st=time.time()

    for i in range(nb_iters):

        input_img_batch, label_img_batch = [], []
        
        for j in range(args.batch_size): # Collect a batch of images

            id = id_list[ i*args.batch_size + j ]

            input_img = data_tool.load_image( dataset_file_name['training']['input'][id] )
            label_img = data_tool.load_image( dataset_file_name['training']['output'][id] )

            with tf.device('/cpu:0'):

                #input_img, label_img = data_tool.data_augmentation( args, input_img, label_img )
                input_img, label_img = data_tool.crop_image_and_label( input_img, label_img, input_size )

                input_img_batch.append( np.expand_dims( np.float32( input_img ) / 255.0 , axis=0) )
                label_img_batch.append( np.expand_dims( np.float32( data_tool.rgb_to_onehot( label_img, label_values_list) ) , axis=0) )

        if args.batch_size == 1:
            input_img_batch = input_img_batch[0]
            label_img_batch = label_img_batch[0]
        else:
            input_img_batch = np.squeeze(np.stack(input_img_batch, axis=1))
            label_img_batch = np.squeeze(np.stack(label_img_batch, axis=1))

        _, current = sess.run(
            fetches=[optimizer,loss],
            feed_dict={net_input:input_img_batch,net_output:label_img_batch}
        )

        current_losses.append(current)
        cnt = cnt + args.batch_size

        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            general_tool.LOG(string_print)
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
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_str))

        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []

        for ind in val_indices: # Do the validation on a small set of validation images

            input_img = data_tool.load_image( dataset_file_name['validation']['input'][ind] )
            label_img = data_tool.load_image( dataset_file_name['validation']['output'][ind] )

            input_img, label_img = data_tool.crop_image_and_label( input_img, label_img, input_size )

            plt.figure()
            plt.imshow(input_img) 
            plt.show()

            plt.figure()
            plt.imshow(label_img) 
            plt.show()

            input_img = np.expand_dims( np.float32( input_img ), axis=0) / 255.0

            label_img_code = data_tool.onehot_to_color_code( data_tool.rgb_to_onehot( label_img, label_values_list), label_values_list )

            output_tensor = sess.run(
                network,
                feed_dict={net_input:input_img})

            output_tensor = np.array(output_tensor[0,:,:,:])

            output_img_code = data_tool.onehot_to_color_code( output_tensor, label_values_list )

            output_img = data_tool.onehot_to_rgb( output_tensor, label_values_list )

            plt.figure()
            plt.imshow( output_img ) 
            plt.show()

            accuracy, class_accuracies, prec, rec, f1, iou = model_tool.evaluate_segmentation(pred=output_img_code, label=label_img_code, nb_class=nb_class)

            file_name = general_tool.filepath_to_name( dataset_file_name['validation']['input'][ind] )
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

            file_name = os.path.basename(dataset_file_name['validation']['input'][ind])
            file_name = os.path.splitext(file_name)[0]

            cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(output_img), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(label_img), cv2.COLOR_RGB2BGR))


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
    remain_time=epoch_time*(args.nb_epoch-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    general_tool.LOG(train_time)
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