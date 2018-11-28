# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import sys
import tensorflow as tf
import numpy as np
from random import randint
from scipy.misc import imread, imresize
import os.path

# customer module 
#from imagenet_classes import class_names    # imagenet 
from selected_classes import class_names    # imagenet 
import vgg                                # VGG-16 model
import data_processing as dp                # data input loading

# file path 
saved_model_filepath = "./model/vgg16/vgg.ckpt"    # File path for saving the trained model

# labels 
test_label_file = './data/ILSVRC2012_test_ground_truth.txt'
validation_label_file = './data/ILSVRC2012_validation_ground_truth.txt'

# Initialize parameters
training_epochs = 1    # for testing
batch_size = 16 # due to the OOM after Regularization # 64     # limitation of GTX 1070 (8GB)
nThread    = 16 #24     # # of Dell server's CPU core
input_width = 224   # size of input for VGG model
input_depth = 3     # number of channels in the image
display_step = 100  # show
#output_classes = 1000   # labels/types/classes of input images
output_classes = 80   # labels/types/classes of input images

validation_dataset_size = 103732

start = time.time()   # Start time for initializing dataset and pipelining

# 1. input pipeline
#rawds = dp.create_raw_dataset("validation")
rawds = dp.create_raw_dataset("training")
dataset = dp.add_pipeline(rawds, batch_size, validation_dataset_size, nThread)
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
logits = vgg.build(x, n_classes=output_classes, training=False)
probs = tf.nn.softmax(logits) 

#prediction = vgg.probs
prediction = probs
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define model saver
saver = tf.train.Saver()


def main():
    
    with tf.Session() as sess:
    
        saver.restore(sess, saved_model_filepath)
    
        # event logs for Tensorboard
        accuracy_history = tf.placeholder(tf.float32)
        accuracy_history_summary = tf.summary.scalar('training_history', accuracy_history)
        merged_history = tf.summary.merge_all()
        
        # Launch graph/Initialize session 
        global_start = time.time()
        print("Initializing global variables...")
        
        sess.run(tf.global_variables_initializer())
        
        # File writers for history
        writer_test = tf.summary.FileWriter('./hist/vgg16/test',sess.graph)
        
        print("Time for initializing variables:", "{0:.2f}".format(time.time()-global_start), "sec")
        
        # Starting time for Testing
        FatalError = False
        t1 = time.time()
        log_writer_step = 0    # Total number of steps/batches for writing event logs, e.g., total steps = training_epochs * num_batch
        epoch_accuracy = []
        
        for epoch in range(training_epochs):
            print("\nStarting epoch", epoch)
            epoch_start = time.time()    # Start time for an epoch
            sess.run(dataset_init_op)
            
            num_batch = int(validation_dataset_size//batch_size)
            for i in range(num_batch):
                start = time.time()    # Start time for a batch
                try:                
                    if i%display_step==0:    # accuracy check
                        test_accuracy = sess.run(accuracy)
                        print("epoch:{},batch:{}/{}".format(epoch, i, num_batch), "acc:{0:.4f}".format(test_accuracy), "ex-time: {0:.2f}sec".format(time.time()-start))
                        writer_test.add_summary(sess.run(merged_history, feed_dict={accuracy_history: test_accuracy * 100}), log_writer_step)
                        log_writer_step = log_writer_step + 1
                        epoch_accuracy.append(test_accuracy)
                except Exception as err:
                    print("Range-Error at epoch:{},batch{}/{}".format(epoch, i, num_batch))
                    print(err)
                    FatalError = True
                    break
                    
            print("Epoch:", epoch, ", Average validation accuracy:", np.mean(epoch_accuracy))
            
        # Ending time for learning
        t2 = time.time()
                       
        print("Testing time: " + "{0:.2f}".format(t2-t1) + " sec")
        
        writer_test.close()

if __name__ == "__main__":
    main()
            