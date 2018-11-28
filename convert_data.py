


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




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
import data_processing as dp
import os, os.path, time


import argparse
import sys



import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin




dataset_folder = "D:/Dataset/Imagenet2012/"
training_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_train/"
testing_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_test/"
validation_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_val/"
saved_model_filepath = "./model/vgg16.ckpt"    # File path for saving the trained model


converted_train_data_filepath = "./data/imagenet_train_data.tfrecords"
converted_test_data_filepath = "./data/imagenet_test_data.tfrecords"
converted_val_data_filepath = "./data/imagenet_val_data.tfrecords"



test_label_file = './data/ILSVRC2012_test_ground_truth.txt'
validation_label_file = './data/ILSVRC2012_validation_ground_truth.txt'




testing_dataset_size = len(os.listdir(testing_dataset_folder))
validation_dataset_size = len(os.listdir(validation_dataset_folder))


training_epochs = 10
batch_size = 64
input_width = 224
input_depth = 3
display_step = 100
output_classes = 1000



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_pickle_from_file(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding='bytes')
        else:
            data_dict = pickle.load(f)
    return data_dict


def convert_to_tfrecord(images, labels, output_file, folder_type="training"):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    start = time.time()

    with tf.python_io.TFRecordWriter(output_file) as record_writer:

        for i in range(len(images)):
        
            try:
                with tf.Session() as sess:
                
                    image_path = images[i]
                    
                    if folder_type == "validation":
                        image_path = validation_dataset_folder + images[i]
                    elif folder_type == "testing":
                        image_path = testing_dataset_folder + images[i]
            
                    imarray = dp.decode_image_opencv(image_path)
                    
                    #imarray = dp.decode_image_with_tf(image_path)
                    #image = imread(str(image_path), mode='RGB')
                    
                    if imarray is not None:
                    
                        #imlabel = dp.get_label(labels[i])
                        imlabel = int(labels[i])
                        
                        # Create a feature
                        feature = {'label': _int64_feature(imlabel),
                                   'image': _bytes_feature(imarray.tostring())}
                        
                        # Create an example protocol buffer
                        example = tf.train.Example(features=tf.train.Features(feature=feature))

                        # Serialize to string and write on the file
                        record_writer.write(example.SerializeToString())

            except Exception as err:
                print("Exception: ", err)
            
    end = time.time()
    print("TIme taken: ", end-start, "seconds\n")
            
      
def main():
    
    
    try:
        # testing images
        test_image_names = os.listdir(testing_dataset_folder)

        f = open(test_label_file, 'r')
        testing_GT = f.readlines()
        f.close()
        
        convert_to_tfrecord(test_image_names, testing_GT, converted_test_data_filepath, "testing")
    except Exception as err:
        print("Exception: ", err)
    

    try:
        # validation images
        val_image_names = os.listdir(validation_dataset_folder)
        
        f = open(validation_label_file, 'r')
        validation_GT = f.readlines()
        f.close()

        convert_to_tfrecord(val_image_names, validation_GT, converted_val_data_filepath, "validation")
    except Exception as err:
        print("Exception: ", err)
        
        
    try:
        # training images
        all_training_images, all_training_image_labels = dp.get_training_images()
        
        # Convert to tf.train.Example and write the to TFRecords.
        convert_to_tfrecord(all_training_images, all_training_image_labels, converted_train_data_filepath, "training")
    except Exception as err:
        print("Exception: ", err)

    
    print('Done!')


if __name__ == '__main__':
    main()


    
    
    

  