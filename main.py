from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys

import helpers 
import utils 

import matplotlib.pyplot as plt

sys.path.append("models")
from FC_DenseNet_Tiramisu import build_fc_densenet
from Encoder_Decoder import build_encoder_decoder
from Encoder_Decoder_Skip import build_encoder_decoder_skip

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--is_training', type=str2bool, default=True, help='Whether we are training or testing')
parser.add_argument('--continue_training', type=bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--class_balancing', type=bool, default=False, help='Whether to use class weight balancing')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=352, help='Height of input image to network')
parser.add_argument('--crop_width', type=int, default=480, help='Width of input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Width of input image to network')
parser.add_argument('--num_val_images', type=int, default=10, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=bool, default=False, help='Whether to do horizontal flipping data augmentation')
parser.add_argument('--v_flip', type=bool, default=False, help='Whether to do vertical flipping data augmentation')
parser.add_argument('--model', type=str, default="FC-DenseNet103", help='The model you are using. Currently supports:\
    FC-DenseNet56, FC-DenseNet67, FC-DenseNet103, Encoder-Decoder, Encoder-Decoder-Skip, custom')
args = parser.parse_args()


# Get a list of the training, validation, and testing file paths
def prepare_data(dataset_dir=args.dataset):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names


# Check if model is available
AVAILABLE_MODELS = ["FC-DenseNet56", "FC-DenseNet67", "FC-DenseNet103", "Encoder-Decoder", "Encoder-Decoder-Skip", "custom"]
if args.model not in AVAILABLE_MODELS:
    print("Error: given model is not available. Try these:")
    print(AVAILABLE_MODELS)
    print("Now exiting ...")
    sys.exit()

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = prepare_data()


class_names_list = helpers.get_class_list(os.path.join(args.dataset, "class_list.txt"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(class_names_list)

print("Setting up training procedure ...")
input = tf.placeholder(tf.float32,shape=[None,None,None,3])
output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network = None
if args.model == "FC-DenseNet56" or args.model == "FC-DenseNet67" or args.model == "FC-DenseNet103":
    network = build_fc_densenet(input, preset_model = args.model, num_classes=num_classes)
elif args.model == "Encoder-Decoder":
    network = build_encoder_decoder(input, num_classes)
elif args.model == "Encoder-Decoder-Skip":
    network = build_encoder_decoder_skip(input, num_classes)
elif args.model == "custom":
    network = build_custom(input, num_classes)


# Compute your (unweighted) softmax cross entropy loss
loss = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output)

if args.class_balancing:
    # Your class weights
    class_weights = utils.median_frequency_balancing(os.path.join(args.dataset, "train_labels"))
    # Deduce weights for batch samples based on their true label
    weights = tf.reduce_sum(class_weights * output, axis=1)
    # Apply the weights, relying on broadcasting of the multiplication
    weighted_losses = loss * weights
    loss = weighted_losses

# Reduce the result to get your final loss
loss = tf.reduce_mean(loss)

opt = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training or not args.is_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

avg_scores_per_epoch = []

if args.is_training:

    print("***** Begin training *****")

    avg_loss_per_epoch = []

    # Which validation images doe we want
    val_indices = []
    num_vals = min(args.num_val_images, len(val_input_names))
    for i in range(num_vals):
        ind = random.randint(0, len(val_input_names) - 1)
        val_indices.append(ind)

    # Do the training here
    for epoch in range(0, args.num_epochs):

        current_losses = []

        cnt=0
        id_list = np.random.permutation(len(train_input_names))
        num_iters = int(np.floor(len(id_list) / args.batch_size))

        for i in range(num_iters):
            st=time.time()
            
            input_image_batch = []
            output_image_batch = [] 

            # Collect a batch of images
            for j in range(args.batch_size):
                index = i*args.batch_size + j
                id = id_list[index]
                input_image = cv2.imread(train_input_names[id],-1)
                output_image = cv2.imread(train_output_names[id],-1)

                # Data augmentation
                input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

                if args.h_flip and random.randint(0,1):
                    input_image = cv2.flip(input_image, 1)
                    output_image = cv2.flip(output_image, 1)
                if args.v_flip and random.randint(0,1):
                    input_image = cv2.flip(input_image, 0)
                    output_image = cv2.flip(output_image, 0)

                # Prep the data. Make sure the labels are in one-hot format
                input_image = np.float32(input_image) / 255.0
                output_image = np.float32(helpers.one_hot_it(label=output_image, num_classes=num_classes))
                
                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

            # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****
            # input_image = tf.image.crop_to_bounding_box(input_image, offset_height=0, offset_width=0, 
            #                                               target_height=args.crop_height, target_width=args.crop_width).eval(session=sess)
            # output_image = tf.image.crop_to_bounding_box(output_image, offset_height=0, offset_width=0, 
            #                                               target_height=args.crop_height, target_width=args.crop_width).eval(session=sess)
            # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****

            # memory()
            
            if args.batch_size == 1:
                input_image_batch = input_image_batch[0]
                output_image_batch = output_image_batch[0]
            else:
                input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

            # Do the training
            _,current=sess.run([opt,loss],feed_dict={input:input_image_batch,output:output_image_batch})
            current_losses.append(current)
            cnt = cnt + args.batch_size
            if cnt % 20 == 0:
                string_print = "Epoch = %d Count = %d Current = %.2f Time = %.2f"%(epoch,cnt,current,time.time()-st)
                utils.LOG(string_print)

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)
        
        # Create directories if needed
        if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
            os.makedirs("%s/%04d"%("checkpoints",epoch))

        saver.save(sess,model_checkpoint_name)
        saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoch))


        target=open("%s/%04d/val_scores.txt"%("checkpoints",epoch),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou %s\n" % (class_names_string))

        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []


        # Do the validation on a small set of validation images
        for ind in val_indices:
            
            input_image = np.expand_dims(np.float32(cv2.imread(val_input_names[ind],-1)[:args.crop_height, :args.crop_width]),axis=0)/255.0
            gt = cv2.imread(val_output_names[ind],-1)[:args.crop_height, :args.crop_width]

            st = time.time()

            output_image = sess.run(network,feed_dict={input:input_image})
            

            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            out = output_image
            output_image = helpers.colour_code_segmentation(output_image)

            accuracy = utils.compute_avg_accuracy(out, gt)
            class_accuracies = utils.compute_class_accuracies(out, gt)
            prec = utils.precision(out[:,:,0], gt)
            rec = utils.recall(out[:,:,0], gt)
            f1 = utils.f1score(out[:,:,0], gt)
            iou = utils.compute_mean_iou(out[:,:,0], gt)
        
            file_name = utils.filepath_to_name(val_input_names[ind])
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
            
            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt))
            gt = helpers.colour_code_segmentation(gt)
 
            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),np.uint8(output_image))
            cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),np.uint8(gt))


        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)

        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

        scores_list = []

    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)

    
    ax1.plot(range(num_epochs), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")


    plt.savefig('accuracy_vs_epochs.png')

    plt.clf()

    ax1 = fig.add_subplot(111)

    
    ax1.plot(range(num_epochs), avg_loss_per_epoch)
    ax1.set_title("Average loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current loss")

    plt.savefig('loss_vs_epochs.png')

else:
    print("***** Begin testing *****")

    # Create directories if needed
    if not os.path.isdir("%s"%("Test")):
            os.makedirs("%s"%("Test"))

    target=open("%s/test_scores.txt"%("Test"),'w')
    target.write("test_name, avg_accuracy, precision, recall, f1 score, mean iou %s\n" % (class_names_string))
    scores_list = []
    class_scores_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []

    # Run testing on ALL test images
    for ind in range(len(test_input_names)):
        input_image = np.expand_dims(np.float32(cv2.imread(test_input_names[ind],-1)[:args.crop_height, :args.crop_width]),axis=0)/255.0
        st = time.time()
        output_image = sess.run(network,feed_dict={input:input_image})
        

        gt = cv2.imread(test_output_names[ind],-1)[:args.crop_height, :args.crop_width]

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)
        out = output_image
        output_image = helpers.colour_code_segmentation(output_image)

        accuracy = utils.compute_avg_accuracy(out, gt)
        class_accuracies = utils.compute_class_accuracies(out, gt)
        prec = utils.precision(out[:,:,0], gt)
        rec = utils.recall(out[:,:,0], gt)
        f1 = utils.f1score(out[:,:,0], gt)
        iou = utils.compute_mean_iou(out[:,:,0], gt)
    
        file_name = utils.filepath_to_name(val_input_names[ind])
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
    
        gt = helpers.reverse_one_hot(helpers.one_hot_it(gt))
        gt = helpers.colour_code_segmentation(gt)

        cv2.imwrite("%s/%s_pred.png"%("Test", file_name),np.uint8(output_image))
        cv2.imwrite("%s/%s_gt.png"%("Test", file_name),np.uint8(gt))


    target.close()

    avg_score = np.mean(scores_list)
    class_avg_scores = np.mean(class_scores_list, axis=0)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_iou = np.mean(iou_list)
    print("Average test accuracy = ", avg_score)
    print("Average per class test accuracies = \n")
    for index, item in enumerate(class_avg_scores):
        print("%s = %f" % (class_names_list[index], item))
    print("Average precision = ", avg_precision)
    print("Average recall = ", avg_recall)
    print("Average F1 score = ", avg_f1)
    print("Average mean IoU score = ", avg_iou)


