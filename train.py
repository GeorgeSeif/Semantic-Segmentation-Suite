import os, time, cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import numpy as np
import random
from os.path import basename, normpath
from sacred import Experiment

# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
matplotlib.use('Agg')

from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt

 # python train.py -m ec2-34-204-179-110.compute-1.amazonaws.com:27017:semseg \
 # with 'args={"num_epochs": 3000, \
 #        "epoch_start_i": 0, \
 #        "checkpoint_step": 5, \
 #        "validation_step": 1, \
 #        "image": None, \
 #        "continue_training": False, \
 #        "dataset": "../datasets/unreal_rugs_sss", \
 #        "crop_height": 512, \
 #        "crop_width": 512, \
 #        "batch_size": 8, \
 #        "num_val_images": 20, \
 #        "h_flip": True, \
 #        "v_flip": True, \
 #        "brightness": 0.3, \
 #        "rotation": 0.1, \
 #        "model": "DeepLabV3_plus", \
 #        "frontend": "ResNet152"}'

ex = Experiment("SemanticSegmentation")

@ex.config
def sem_seg_config():
    args = {
        "num_epochs": 3000,
        "epoch_start_i": 0,
        "checkpoint_step": 5,
        "validation_step": 1,
        "image": None,
        "continue_training": False,
        "dataset": "../datasets/ade20k_sss",
        "crop_height": 512,
        "crop_width": 512,
        "batch_size": 8,
        "num_val_images": 20,
        "h_flip": False,
        "v_flip": False,
        "brightness": None,
        "rotation": None,
        "model": "DeepLabV3_plus",
        "frontend": "ResNet152"
    }

@ex.automain
def main(args, _log):
    def data_augmentation(input_image, output_image):
        # Data augmentation
        input_image, output_image = utils.random_crop(input_image, output_image, args["crop_height"], args["crop_width"])

        if args["h_flip"] and random.randint(0,1):
            input_image = cv2.flip(input_image, 1)
            output_image = cv2.flip(output_image, 1)
        if args["v_flip"] and random.randint(0,1):
            input_image = cv2.flip(input_image, 0)
            output_image = cv2.flip(output_image, 0)
        if args["brightness"]:
            factor = 1.0 + random.uniform(-1.0*args["brightness"], args["brightness"])
            table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            input_image = cv2.LUT(input_image, table)
        if args["rotation"]:
            angle = random.uniform(-1*args["rotation"], args["rotation"])
        if args["rotation"]:
            M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
            input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
            output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

        return input_image, output_image

    print("Args:", args)

    # Get the names of the classes so we can record the evaluation results
    class_names_list, label_values = helpers.get_label_info(os.path.join(args["dataset"], "class_dict.csv"))
    class_names_string = ""
    for class_name in class_names_list:
        if not class_name == class_names_list[-1]:
            class_names_string = class_names_string + class_name + ", "
        else:
            class_names_string = class_names_string + class_name

    num_classes = len(label_values)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)


    # Compute your softmax cross entropy loss
    net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
    net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

    network, init_fn = model_builder.build_model(model_name=args["model"], frontend=args["frontend"], net_input=net_input, num_classes=num_classes, crop_width=args["crop_width"], crop_height=args["crop_height"], is_training=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))

    opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])

    saver=tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    utils.count_params()

    # If a pre-trained ResNet is required, load the weights.
    # This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
    if init_fn is not None:
        init_fn(sess)

    # Load a previous checkpoint if desired
    model_checkpoint_name = "checkpoints/latest_model_" + args["model"] + "_" + basename(normpath(args["dataset"])) + ".ckpt"
    if args["continue_training"]:
        _log.info('Loaded latest model checkpoint')
        saver.restore(sess, model_checkpoint_name)

    # Load the data
    _log.info("Loading the data ...")
    train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args["dataset"])



    _log.info("\n***** Begin training *****")
    _log.debug("Dataset -->", args["dataset"])
    _log.debug("Model -->", args["model"])
    _log.debug("Crop Height -->", args["crop_height"])
    _log.debug("Crop Width -->", args["crop_width"])
    _log.debug("Num Epochs -->", args["num_epochs"])
    _log.debug("Batch Size -->", args["batch_size"])
    _log.debug("Num Classes -->", num_classes)

    _log.debug("Data Augmentation:")
    _log.debug("\tVertical Flip -->", args["v_flip"])
    _log.debug("\tHorizontal Flip -->", args["h_flip"])
    _log.debug("\tBrightness Alteration -->", args["brightness"])
    _log.debug("\tRotation -->", args["rotation"])

    avg_loss_per_epoch = []
    avg_scores_per_epoch = []
    avg_iou_per_epoch = []

    # Which validation images do we want
    val_indices = []
    num_vals = min(args["num_val_images"], len(val_input_names))

    # Set random seed to make sure models are validated on the same validation images.
    # So you can compare the results of different models more intuitively.
    random.seed(16)
    val_indices=random.sample(range(0,len(val_input_names)),num_vals)

    # Do the training here
    for epoch in range(args["epoch_start_i"], args["num_epochs"]):

        current_losses = []

        cnt=0

        # Equivalent to shuffling
        id_list = np.random.permutation(len(train_input_names))

        num_iters = int(np.floor(len(id_list) / args["batch_size"]))
        st = time.time()
        epoch_st=time.time()
        for i in range(num_iters):
            # st=time.time()

            input_image_batch = []
            output_image_batch = []

            # Collect a batch of images
            for j in range(args["batch_size"]):
                index = i*args["batch_size"] + j
                id = id_list[index]
                input_image = utils.load_image(train_input_names[id], args["crop_width"], args["crop_height"])
                output_image = utils.load_image(train_output_names[id], args["crop_width"], args["crop_height"])

                with tf.device('/cpu:0'):
                    input_image, output_image = data_augmentation(input_image, output_image)


                    # Prep the data. Make sure the labels are in one-hot format
                    input_image = np.float32(input_image) / 255.0
                    output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

                    input_image_batch.append(np.expand_dims(input_image, axis=0))
                    output_image_batch.append(np.expand_dims(output_image, axis=0))

            if args["batch_size"] == 1:
                input_image_batch = input_image_batch[0]
                output_image_batch = output_image_batch[0]
            else:
                input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

            # Do the training
            _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
            current_losses.append(current)
            cnt = cnt + args["batch_size"]
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
        _log.info("Saving latest checkpoint")
        saver.save(sess,model_checkpoint_name)

        if val_indices != 0 and epoch % args["checkpoint_step"] == 0:
            _log.info("Saving checkpoint for this epoch")
            saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoch))


        if epoch % args["validation_step"] == 0:
            _log.info("Performing validation")
            target_path = "%s/%04d/val_scores.csv" % ("checkpoints", epoch)
            target=open(target_path,'w')
            target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
            

            scores_list = []
            class_scores_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            iou_list = []


            # Do the validation on a small set of validation images
            for ind in val_indices:

                input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind], args["crop_width"], args["crop_height"])[:args["crop_height"], :args["crop_width"]]),axis=0)/255.0
                gt = utils.load_image(val_output_names[ind], args["crop_width"], args["crop_height"])[:args["crop_height"], :args["crop_width"]]
                gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

                # st = time.time()

                output_image = sess.run(network,feed_dict={net_input:input_image})


                output_image = np.array(output_image[0,:,:,:])
                output_image = helpers.reverse_one_hot(output_image)
                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

                accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

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

                gt = helpers.colour_code_segmentation(gt, label_values)

                file_name = os.path.basename(val_input_names[ind])
                file_name = os.path.splitext(file_name)[0]

                pred_img_path = "%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name)
                gt_img_path = "%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name)

                cv2.imwrite(pred_img_path, cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite(gt_img_path, cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
                
                # Send the first 16 images to sacred
                if ind in val_indices[:16]:
                    ex.add_artifact(gt_img_path, "GtImage_%d" % ind)
                    ex.add_artifact(pred_img_path, "PredImage_%d" % ind)

            target.close()
            ex.add_artifact(target_path)

            avg_score = np.mean(scores_list)
            class_avg_scores = np.mean(class_scores_list, axis=0)
            avg_scores_per_epoch.append(avg_score)
            avg_precision = np.mean(precision_list)
            avg_recall = np.mean(recall_list)
            avg_f1 = np.mean(f1_list)
            avg_iou = np.mean(iou_list)
            avg_iou_per_epoch.append(avg_iou)

            # Sacred info dict gets sent every heartbeat (10s)
            ex.info["avg_score"] = avg_score
            ex.info["class_avg_scores"] = class_avg_scores
            ex.info["avg_precision"] = avg_precision
            ex.info["avg_recall"] = avg_recall
            ex.info["avg_f1"] = avg_f1
            ex.info["avg_iou"] = avg_iou

            _log.debug("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
            _log.debug("Average per class validation accuracies for epoch # %04d:"% (epoch))
            for index, item in enumerate(class_avg_scores):
                _log.debug("%s = %f" % (class_names_list[index], item))
            _log.debug("Validation precision = ", avg_precision)
            _log.debug("Validation recall = ", avg_recall)
            _log.debug("Validation F1 score = ", avg_f1)
            _log.debug("Validation IoU score = ", avg_iou)

        epoch_time=time.time()-epoch_st
        remain_time=epoch_time*(args["num_epochs"]-1-epoch)
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
        ex.add_artifact("accuracy_vs_epochs.png")

        plt.clf()

        fig2, ax2 = plt.subplots(figsize=(11, 8))

        ax2.plot(range(epoch+1), avg_loss_per_epoch)
        ax2.set_title("Average loss vs epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Current loss")

        plt.savefig('loss_vs_epochs.png')
        ex.add_artifact("loss_vs_epochs.png")

        plt.clf()

        fig3, ax3 = plt.subplots(figsize=(11, 8))

        ax3.plot(range(epoch+1), avg_iou_per_epoch)
        ax3.set_title("Average IoU vs epochs")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Current IoU")

        plt.savefig('iou_vs_epochs.png')
        ex.add_artifact("iou_vs_epochs.png")


