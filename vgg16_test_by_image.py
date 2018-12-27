# Hide the warning messages about CPU/GPU
from selected_classes import class_names    # imagenet
import data_processing as dp                # data input loading
import vgg                                # VGG-16 model
import os.path
from scipy.misc import imread, imresize
from random import randint
import numpy as np
import tensorflow as tf
import sys
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# customer module
# from imagenet_classes import class_names    # imagenet

# file path
# File path for saving the trained model
saved_model_filepath = "./model/vgg16/vgg.ckpt"

input_width = 224   # size of input for VGG model
input_depth = 3     # number of channels in the image
# output_classes = 1000   # labels/types/classes of input images
output_classes = 80   # labels/types/classes of input images

# Initialize input and output
x = tf.placeholder(tf.float32, shape=[
                   None, input_width, input_width, input_depth])
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
                # print(prob)
                preds = (np.argsort(prob)[::-1])[0:5]
                print(preds)
                for p in preds:
                    print(class_names[p], prob[p])
            except Exception as err:
                print(err)


if __name__ == "__main__":
    main(sys.argv)
