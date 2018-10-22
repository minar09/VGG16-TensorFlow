

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import time
import tensorflow as tf
import numpy as np
from random import randint
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import vgg16
import os, os.path, time




def getNumber(label):
    for i, name in range(class_names):
        if name == label or label in name:
            return i
            
            
def get_label(label):
    return np.eye(output_classes, dtype=np.float32)[int(label)]
    
            
def decode_image(pathToImage):
    image = imread(str(pathToImage), mode='RGB')
    imarray = imresize(image, (input_width, input_width))
    imarray = imarray.reshape(input_width, input_width, input_depth)
    
    return imarray
    
    
def decode_image_with_tf(pathToImage):
    imageContents = tf.read_file(str(pathToImage))
    image = tf.image.decode_jpeg(imageContents, channels=3)
    resized_image = tf.image.resize_images(image, [input_width, input_width])
    imarray = resized_image.eval()
    imarray = imarray.reshape(input_width, input_width, input_depth)
    
    return imarray
            

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)     
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width]) 
    return resized_image.eval()
    
    
def training_preprocess(image):
    crop_image = tf.random_crop(image, [input_width, input_width, input_depth])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)          

    centered_image = flip_image - means                               

    return centered_image
    
    
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, input_width, input_width)  

    centered_image = crop_image - means                                    

    return centered_image, label
    
    
