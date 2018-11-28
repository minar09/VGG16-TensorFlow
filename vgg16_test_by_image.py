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

input_width = 224   # size of input for VGG model
input_depth = 3     # number of channels in the image
#output_classes = 1000   # labels/types/classes of input images
output_classes = 80   # labels/types/classes of input images

# Initialize input and output
x = tf.placeholder(tf.float32, shape=[None, input_width, input_width, input_depth])
y = tf.placeholder(tf.float32, shape=[None, output_classes])

# 2. classifier model
# @TODO use stored weight and session  See  def __init__(self, imgs, weights=None, sess=None):
logits = vgg.build(x, n_classes=output_classes, training=False)
probs = tf.nn.softmax(logits) 

# Define model saver
saver = tf.train.Saver()


def main(argv):
    
    with tf.Session() as sess:
    
        saver.restore(sess, saved_model_filepath)
    
        if(argv):
            try:
                argv = argv[1]
        
                #imgs = tf.placeholder(tf.float32, [None, input_width, input_width, input_depth])
                #vgg = vgg11.vgg11(imgs, saved_model_filepath, sess)

                img1 = dp.decode_image_opencv(argv)
                #image = dp._parse_function(argv)
                #img1 = dp.training_preprocess(image)

                prob = sess.run(probs, feed_dict={x: [img1]})[0]
                #print(prob)
                preds = (np.argsort(prob)[::-1])[0:5]
                print(preds)
                for p in preds:
                    print(class_names[p], prob[p])
            except Exception as err:
                print(err)
                
    
if __name__ == "__main__":
    main(sys.argv)
            