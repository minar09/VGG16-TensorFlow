# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import tensorflow as tf
import numpy as np
from random import randint
from scipy.misc import imread, imresize
import os.path

# customer module 
from imagenet_classes import class_names    # imagenet 
import vgg11                                # VGG-11 model
import data_processing as dp                # data input loading

# dataset path 
dataset_folder = "D:/Dataset/Imagenet2012/"
training_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_train/"
testing_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_test/"
validation_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_val/"
saved_model_filepath = "./model/vgg11/vgg.ckpt"    # File path for saving the trained model

# labels 
test_label_file = './data/ILSVRC2012_test_ground_truth.txt'
validation_label_file = './data/ILSVRC2012_validation_ground_truth.txt'

#training_dataset_size = 1281167
training_dataset_size = 32
testing_dataset_size = 100000
validation_dataset_size = 50000

# Initialize parameters
training_epochs = 2 # for testing
batch_size = 32 # due to the OOM after Regularization # 64     # limitation of GTX 1070 (8GB)
nThread    = 16 #24     # # of Dell server's CPU core
input_width = 224   # size of input for VGG model
input_depth = 3     # number of channels in the image
display_step = 10  # show
output_classes = 1000   # labels/types/classes of input images

learning_rate = 0.01   # could be initial learning rate if decreased exponentially

start = time.time()   # Start time for initializing dataset and pipelining

# 1. input pipeline
rawds = dp.create_raw_dataset("training")
dataset = dp.add_pipeline(rawds, batch_size, training_dataset_size, nThread)
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
dataset_init_op = iterator.make_initializer(dataset)
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

print(images.shape, labels.shape)
print("Time for initializing dataset and pipelining:", "{0:.2f}".format(time.time()-start), "sec")
       
# Initialize input and output
x = images      # image batch 
y = labels      # label batch 

# 2. classifier model
# @TODO use stored weight and session  See  def __init__(self, imgs, weights=None, sess=None):
vgg = vgg11.vgg11(x)
logits = vgg.logits

# 3. Loss & optimizer function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

prediction = vgg.probs
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define model saver
saver = tf.train.Saver()


if __name__ == '__main__':

    # event logs for Tensorboard
    accuracy_history = tf.placeholder(tf.float32)
    accuracy_history_summary = tf.summary.scalar('training_history', accuracy_history)
    merged_history = tf.summary.merge_all()

    # Launch graph/Initialize session 
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        global_start = time.time()
        print("Initializing global variables...")
        
        sess.run(tf.global_variables_initializer())
        
        # File writers for history
        writer_train = tf.summary.FileWriter('./hist/vgg11/train',sess.graph)
        writer_loss = tf.summary.FileWriter('./hist/vgg11/loss',sess.graph)
        
        # @TODO  restore parameters !!
        print("Time for initializing variables:", "{0:.2f}".format(time.time()-global_start), "sec")
        
        # Restore parameters from trained model
        ts = time.time()
        try:
            #saver.restore(sess, saved_model_filepath)
            # load the stored file 
            ckpt = tf.train.get_checkpoint_state('./model/vgg11')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):  
                print("restoring from ", ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("no parameter file exists!")
        except Exception as err:
            print("Error in restoring model: ", err)
            
        print("Time for restoring params", "{0:.2f}".format(time.time()-ts), "sec")

        # Starting time for learning
        FatalError = False
        t1 = time.time()
        log_writer_step = 0    # Total number of steps/batches for writing event logs, e.g., total steps = training_epochs * num_batch
        
        print(sess.run(vgg.fc1w))
        
        for epoch in range(training_epochs):
            print("\nStarting epoch", epoch)
            epoch_start = time.time()    # Start time for an epoch
            sess.run(dataset_init_op)
            
            num_batch = int(training_dataset_size//batch_size)
            for i in range(num_batch):
                start = time.time()    # Start time for a batch
                try:                
                    if i%display_step:
                        _, loss_print = sess.run([optimizer, loss])  # training \
                    else: # accuracy check
                        _, loss_print, train_accuracy = sess.run([optimizer, loss, accuracy])
                        print("epoch:{},batch:{}/{}".format(epoch, i, num_batch), "loss:{:.4f}".format(loss_print), "acc:{0:.4f}".format(train_accuracy), "ex-time: {0:.2f}sec".format(time.time()-start))
                        writer_loss.add_summary(sess.run(merged_history, feed_dict={accuracy_history: loss_print}), log_writer_step)
                        writer_train.add_summary(sess.run(merged_history, feed_dict={accuracy_history: train_accuracy * 100}), log_writer_step)
                        log_writer_step = log_writer_step + 1
                except Exception as err:
                    print("Range-Error at epoch:{},batch{}/{}".format(epoch, i, num_batch))
                    print(err)
                    FatalError = True
                    break
                   
            # Save trained model after an epoch
            savedPath = saver.save(sess, saved_model_filepath)
            print("Model saved at: " , savedPath, ", after epoch", epoch)
            if FatalError:
                break
                
        print(sess.run(vgg.fc1w))
            
        #label .failure 
        savedPath = saver.save(sess, saved_model_filepath)
        print("Model saved at: " , savedPath, ", after epoch", epoch)
        print("\n")
        # Ending time for learning
        t2 = time.time()
                       
        print("Learning time: " + "{0:.2f}".format(t2-t1) + " sec")
        
        writer_train.close()
        writer_loss.close()
      
print("\nTraining finished!")


