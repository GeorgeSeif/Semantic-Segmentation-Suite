from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime

from helper import *

from FC_DenseNet_Tiramisu import build_fc_densenet

def LOG(X, f=None):
	time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	if not f:
		print(time_stamp + " " + X)
	else:
		f.write(time_stamp + " " + X)

def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
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
    	val_input_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
    	cwd = os.getcwd()
    	val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
    	cwd = os.getcwd()
    	test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
    	cwd = os.getcwd()
    	test_output_names.append(cwd + dataset_dir + "/test_labels/" + file)
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names

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
batch_size=4
num_epochs=10

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

print("***** Begin training *****")

if is_training:

    for epoch in range(1,num_epochs):
        
        input_image_names=[None]*len(train_input_names)
        output_image_names=[None]*len(train_input_names)

        cnt=0
        for id in np.random.permutation(len(train_input_names)):
            st=time.time()
            if input_image_names[id] is None:
            	LOG(train_input_names[id])
                input_image_names[id] = train_input_names[id]
                output_image_names[id] = train_output_names[id]
                input_image=np.expand_dims(np.float32(cv2.imread(input_image_names[id],-1)[:352, :480]),axis=0)/255.0
                output_image=np.expand_dims(np.float32(one_hot_it(labels=cv2.imread(output_image_names[id],-1)[:352, :480])), axis=0)

                # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****
                # input_image = tf.image.crop_to_bounding_box(input_image, offset_height=0, offset_width=0, 
                # 												target_height=352, target_width=480).eval(session=sess)
                # output_image = tf.image.crop_to_bounding_box(output_image, offset_height=0, offset_width=0, 
                # 												target_height=352, target_width=480).eval(session=sess)
                # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****
                if input_image.shape[1]*input_image.shape[2]>2200000:#due to GPU memory limitation
                    print("Skipping due to GPU memory limitation")
                    continue

                memory()
            
                _,current=sess.run([opt,loss],feed_dict={input:input_image,output:output_image})
                cnt = cnt + 1
                string_print = "Epoch = %d Count = %d Current = %.2f Time = %.2f"%(epoch,cnt,current,time.time()-st)
                LOG(string_print)

        # os.makedirs("%s/%04d"%(task,epoch))
        # target=open("%s/%04d/score.txt"%(task,epoch),'w')
        # target.write("%f"%np.mean(all[np.where(all)]))
        # target.close()

        # saver.save(sess,"%s/model.ckpt"%task)
        # saver.save(sess,"%s/%04d/model.ckpt"%(task,epoch))
        # for ind in range(10):
        #     input_image=np.expand_dims(np.float32(cv2.imread(val_names[ind],-1)),axis=0)/255.0
        #     st=time.time()
        #     output_image=sess.run(network,feed_dict={input:input_image})
        #     print("%.3f"%(time.time()-st))
        #     output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
        #     cv2.imwrite("%s/%04d/%06d.jpg"%(task,epoch,ind+1),np.uint8(output_image[0,:,:,:]))




# -- Implement batch training
# -- Save checkpoints
# -- Run on validation set during training
# -- Show outputs properly and save them
# -- Implement test functionality
# -- Allow for using Citiscapes dataset
# -- Implement the 100 layer version
