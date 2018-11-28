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
import vgg16                                # VGG-16 model
import data_processing as dp                # data input loading
import TowerProcessing as tp                # Tower functions for parallelism using multiple GPUs

# dataset path 
dataset_folder = "D:/Dataset/Imagenet2012/"
training_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_train/"
testing_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_test/"
validation_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_val/"
saved_model_filepath = "./model/vgg16.ckpt"    # File path for saving the trained model

# labels 
test_label_file = './data/ILSVRC2012_test_ground_truth.txt'
validation_label_file = './data/ILSVRC2012_validation_ground_truth.txt'

training_dataset_size = 1281167
testing_dataset_size = 100000
validation_dataset_size = 50000

# Initialize parameters
training_epochs = 10
batch_size = 64     # limitation of GTX 1070 (8GB)
nThread    = 16 #24     # # of Dell server's CPU core
input_width = 224   # size of input for VGG model
input_depth = 3     # number of channels in the image
display_step = 100  # show
output_classes = 1000   # labels/types/classes of input images
num_gpus = 2        # Number of GPUs for parallelism

learning_rate = 0.001   # could be initial learning rate if decreased exponentially
weight_decay = 0.0005

start = time.time()   # Start time for initializing dataset and pipelining

# 1. input pipeline
rawds = dp.create_raw_dataset("training")
dataset = dp.add_pipeline(rawds, batch_size, training_dataset_size, nThread)
#iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
#dataset_init_op = iterator.make_initializer(dataset)
#next_element = iterator.get_next()
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

print(images.shape, labels.shape)
print("Time for initializing dataset and pipelining:", "{0:.2f}".format(time.time()-start), "seconds")
       
# Initialize input and output
#x = tf.placeholder(tf.float32, shape=[None, input_width, input_width, input_depth])
#y = tf.placeholder(tf.float32, shape=[None, output_classes])
# dataset iterator = to input  
x = images      # image batch 
y = labels      # label batch 

# 2. classifier model
# @TODO use stored weight and session  See  def __init__(self, imgs, weights=None, sess=None):
vgg = vgg16.vgg16(x)
logits = vgg.logits

# 3. Loss & optimizer function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
#regularizer = tf.nn.l2_loss(vgg.fc1w) + tf.nn.l2_loss(vgg.fc2w)
#loss = tf.reduce_mean(loss + regularizer * weight_decay)
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
    with tf.Graph().as_default(),  tf.device('/cpu:0'):
    
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        global_start = time.time()
        print("Initializing global variables...")
        
        sess.run(tf.global_variables_initializer())
        
        # File writers for history
        writer_train = tf.summary.FileWriter('./model/hist/train',sess.graph)
        writer_test = tf.summary.FileWriter('./model/hist/test',sess.graph)
        writer_loss = tf.summary.FileWriter('./model/hist/loss',sess.graph)
        
        # @TODO  restore parameters !!
        print("Time needed for initializing variables:", "{0:.2f}".format(time.time()-global_start), "sec")
        
        # Restore parameters from trained model
        try:
            saver.restore(sess, saved_model_filepath)
        except Exception as err:
            print("Error in restoring model: ", err)

        # Starting time for learning
        t1 = time.time()
        log_writer_step = 0    # Total number of steps/batches for writing event logs, e.g., total steps = training_epochs * num_batch
        for epoch in range(training_epochs):
            print("\nStarting epoch", epoch + 1)
            epoch_start = time.time()    # Start time for an epoch
            #sess.run(dataset_init_op)
            num_batch = int(training_dataset_size//(batch_size * num_gpus))
            for i in range(num_batch):
                start = time.time()    # Start time for a batch
                try:
                    # Calculate the gradients for each model tower.
                    tower_grads = []
                    with tf.variable_scope(tf.get_variable_scope()):
                      for i in xrange(FLAGS.num_gpus):
                        with tf.device('/gpu:%d' % i):
                          with tf.name_scope('tower_%d' % i) as scope:
                            # Calculate the loss for one tower of the VGG model. This function
                            # constructs the entire VGG model but shares the variables across
                            # all towers.
                            #loss = tower_loss(scope, image_batch, label_batch)
                            _, loss_print = sess.run([optimizer, loss])

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Retain the summaries from the final tower.
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                            # Calculate the gradients for the batch of data on this VGG tower.
                            grads = opt.compute_gradients(loss)

                            # Keep track of the gradients across all towers.
                            tower_grads.append(grads)

                    # We must calculate the mean of each gradient. Note that this is the
                    # synchronization point across all towers.
                    grads = tp.average_gradients(tower_grads)                

                    if i%display_step:
                        _, loss_print = sess.run([optimizer, loss])  # training \
                        print("epoch:", epoch+1, "batch:", i, "/",num_batch, ", loss:", loss_print, ", ex-time:", "{0:.2f}".format(time.time()-start), "sec")
                        writer_loss.add_summary(sess.run(merged_history, feed_dict={accuracy_history: loss_print}), log_writer_step)
                    else: # accuracy check                     
                        loss_print, train_accuracy = sess.run([loss, accuracy])
                        print("epoch:", epoch+1, ", batch:",i, "/",num_batch, ", loss:", loss_print, ", acc:", train_accuracy, "{0:.2f}".format(time.time()-start), "sec")
                        writer_loss.add_summary(sess.run(merged_history, feed_dict={accuracy_history: loss_print}), log_writer_step)
                        writer_train.add_summary(sess.run(merged_history, feed_dict={accuracy_history: train_accuracy}), log_writer_step)
                except Exception as err:
                    print(err)
                log_writer_step = log_writer_step + 1
                
            # Save trained model after an epoch
            savedPath = saver.save(sess, saved_model_filepath)
            print("Model saved at: " , savedPath, ", after epoch", epoch)

        print("\n")
        # Ending time for learning
        t2 = time.time()
                
        # Save trained model
        savedPath = saver.save(sess, saved_model_filepath)
        print("Final model saved at: " , savedPath)
            
        print("Learning time: " + "{0:.2f}".format(t2-t1) + " sec")
        
print("\nTraining finished!")


