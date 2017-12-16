from __future__ import print_function
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime

from helper import *

import matplotlib.pyplot as plt

from FC_DenseNet_Tiramisu import build_fc_densenet

def LOG(X, f=None):
	time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	if not f:
		print(time_stamp + " " + X)
	else:
		f.write(time_stamp + " " + X)

def compute_accuracy(y_pred, y_true):
    # print(y_true.shape)
    w = y_true.shape[0]
    h = y_true.shape[1]
    total = w*h
    count = 0.0
    for i in range(w):
        for j in range(h):
            if y_pred[i, j] == y_true[i, j]:
                count = count + 1.0
    # print(count)
    return count / total


def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB
    print('memory use:', memoryUse)

def prepare_data(dataset_dir="CamVid"):
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
    	test_output_names.append(cwd + dataset_dir + "/test_labels/" + file)
    return train_input_names[:10],train_output_names[:10], val_input_names, val_output_names, test_input_names, test_output_names

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = prepare_data()


print("Setting up training procedure ...")
input=tf.placeholder(tf.float32,shape=[None,None,None,3])
output=tf.placeholder(tf.float32,shape=[None,None,None,12])
network=build_fc_densenet(input)



loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output))

opt=tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])

is_training=True
num_epochs=3


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

avg_scores_per_epoch = []

print("***** Begin training *****")

if is_training:

    for epoch in range(num_epochs):
        
        input_image_names=[None]*len(train_input_names)
        output_image_names=[None]*len(train_input_names)

        cnt=0
        for id in np.random.permutation(len(train_input_names)):
            st=time.time()
            if input_image_names[id] is None:
            	# LOG(train_input_names[id])
                input_image_names[id] = train_input_names[id]
                output_image_names[id] = train_output_names[id]
                input_image = np.expand_dims(np.float32(cv2.imread(input_image_names[id],-1)[:352, :480]),axis=0)/255.0
                output_image = np.expand_dims(np.float32(one_hot_it(labels=cv2.imread(output_image_names[id],-1)[:352, :480])), axis=0)

                # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****
                # input_image = tf.image.crop_to_bounding_box(input_image, offset_height=0, offset_width=0, 
                # 												target_height=352, target_width=480).eval(session=sess)
                # output_image = tf.image.crop_to_bounding_box(output_image, offset_height=0, offset_width=0, 
                # 												target_height=352, target_width=480).eval(session=sess)
                # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****

                if input_image.shape[1]*input_image.shape[2]>2200000:#due to GPU memory limitation
                    print("Skipping due to GPU memory limitation")
                    continue

                # memory()
            
                _,current=sess.run([opt,loss],feed_dict={input:input_image,output:output_image})
                cnt = cnt + 1
                string_print = "Epoch = %d Count = %d Current = %.2f Time = %.2f\r"%(epoch,cnt,current,time.time()-st)
                print(string_print)


        # print("%.3f"%(time.time()-st))
        
        if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
            os.makedirs("%s/%04d"%("checkpoints",epoch))

        saver.save(sess,"%s/final_model.ckpt"%"checkpoints")
        saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoch))


        target=open("%s/%04d/scores.txt"%("checkpoints",epoch),'w')
        target.write("val_index, accuracy\n")
        val_indices = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
        scores_list = []

        for ind in val_indices:
            input_image = np.expand_dims(np.float32(cv2.imread(val_input_names[ind],-1)[:352, :480]),axis=0)/255.0
            st = time.time()
            output_image = sess.run(network,feed_dict={input:input_image})
            # print("%.3f"%(time.time()-st))
            

            output_image = np.array(output_image[0,:,:,:])
            output_image = reverse_one_hot(output_image)
            out = output_image
            output_image = colour_code_segmentation(output_image)

            gt = cv2.imread(val_output_names[ind],-1)[:352, :480]

            accuracy = compute_accuracy(out, gt)
            target.write("%d, %f\n"%(ind, accuracy))
            print("Accuracy = ", accuracy)

            scores_list.append(accuracy)
            # print(gt.shape)
            gt = colour_code_segmentation(np.expand_dims(gt, axis=-1))
 
            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),np.uint8(output_image))
            cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),np.uint8(gt))


        target.close()

        avg_scores_per_epoch.append(np.mean(scores_list))

        scores_list = []

    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)

    
    ax1.plot(range(num_epochs), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")


    plt.savefig('accuracy_vs_epochs.png')





# -- Load latest checkpoint
# -- Implement test functionality
# -- Allow for using Citiscapes dataset
# -- Implement the 100 layer version
# -- Update comments and clean-up code