

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



# Initialize parameters
training_epochs = 10
batch_size = 64
input_width = 224
input_depth = 3
display_step = 100
output_classes = 1000


learning_rate = 0.001
VGG_MEAN = [123.68, 116.779, 103.939]
means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
weight_decay = 0.0005




def getNumber(label):
    for i, name in range(class_names):
        if name == label or label in name:
            return i
            
            
def get_label(label):
    return np.eye(output_classes, dtype=np.float32)[int(label)]
    
    
            
def decode_image_opencv(pathToImage):

    try:
        image = imread(str(pathToImage), mode='RGB')
        imarray = imresize(image, (input_width, input_width))
        imarray = imarray.reshape(input_width, input_width, input_depth)
        
        return imarray
        
    except Exception as err:
    
        print("Exception: ", err)
        return None
    
    
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
    
    
def shuffle_images_np(imagesPathArray, imagesLabelsArray):
    
    print("Shuffling the images...")
    
    dataset_size = len(imagesPathArray)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    imagesPathArray = np.array(imagesPathArray)[indices.astype(int)]
    imagesPathArray = imagesPathArray[indices]
    
    imagesLabelsArray = np.array(imagesLabelsArray)[indices.astype(int)]
    imagesLabelsArray = imagesLabelsArray[indices]

    return imagesPathArray, imagesLabelsArray
    
    
def shuffle_images_rand(imagesPathArray, imagesLabelsArray):
    
    print("Shuffling the images...")
    
    for i in range(0, len(imagesPathArray)):
        randomIndex1 = randint(0, len(imagesPathArray)-1)
        randomIndex2 = randint(0, len(imagesPathArray)-1)
        imagesPathArray[randomIndex1], imagesPathArray[randomIndex2] = imagesPathArray[randomIndex2], imagesPathArray[randomIndex1]
        imagesLabelsArray[randomIndex1], imagesLabelsArray[randomIndex2] = imagesLabelsArray[randomIndex2], imagesLabelsArray[randomIndex1]
    
    return imagesPathArray, imagesLabelsArray
    
    
def _image_preprocess_fn(image_buffer):
    # image_buffer 1-D string Tensor representing the raw JPEG image buffer.

    # Extract image shape from raw JPEG image buffer.
    image_shape = tf.image.extract_jpeg_shape(image_buffer)

    # Get a crop window with distorted bounding box.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      image_shape, ...)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Decode and crop image.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    cropped_image = tf.image.decode_and_crop_jpeg(image, crop_window)
    
    

def getListOfImages(training_image_folders):
    
    all_training_images = []
    all_training_image_labels = []
    
    for f in range(len(training_image_folders)):
        images = os.listdir(training_dataset_folder + training_image_folders[f])
        for im in range(len(images)):
            image_path = training_dataset_folder + training_image_folders[f] + "/" + images[im]
            all_training_images.append(image_path)
            all_training_image_labels.append(f)
            
    return all_training_images, all_training_image_labels

    
    
def get_training_images():

    training_image_folders = os.listdir(training_dataset_folder)
    for each in training_image_folders:
        if ".tar" in each:
            training_image_folders.remove(each)
    print("Training folders : ", len(training_image_folders))
        
        
    all_training_images, all_training_image_labels = getListOfImages(training_image_folders)

    print("Number of total training images: ", len(all_training_images))
        
    return all_training_images, all_training_image_labels
    
    

        
# Return the batch of training data for next run
def get_next_batch_of_training_images(imagesPathArray, image_labels):

    #start = time.time()
    
    dataset = np.ndarray(shape=(0, input_width, input_width, input_depth), dtype=np.float32)
    labels = np.ndarray(shape=(0, output_classes), dtype=np.float32)

    for i in range(len(imagesPathArray)):

        try:
            
            imarray = decode_image_opencv(imagesPathArray[i])
            
            imlabel = get_label(image_labels[i]) 
            
            appendingImageArray = np.array([imarray], dtype=np.float32)
            
            appendingNumberLabel = np.array([imlabel], dtype=np.float32)
            
            dataset = np.append(dataset, appendingImageArray, axis=0)
            
            labels = np.append(labels, appendingNumberLabel, axis=0)
            
        except Exception as err:
            print("Unexpected image - ", imagesPathArray[i], ", skipping...", image_labels[i] , err)
    
    #end = time.time()
    #print("Getting a batch of 64 images, ", end-start)
    
    return dataset, labels
            
            
            
def get_testing_images(imagesPathArray, labels_array, type="testing"):

    dataset = np.ndarray(shape=(0, input_width, input_width, input_depth), dtype=np.float32)
    labels = np.ndarray(shape=(0, output_classes), dtype=np.float32)
    
    for i in range(len(imagesPathArray)):

        try:
            pathToImage = testing_dataset_folder + imagesPathArray[i]
            if type == "validation":
                pathToImage = validation_dataset_folder + imagesPathArray[i]
                
            imarray = decode_image_opencv(pathToImage)

            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array([getNumber(labels_array[i])], dtype=np.float32)
            
            dataset = np.append(dataset, appendingImageArray, axis=0)
            labels = np.append(labels, appendingNumberLabel, axis=0)
        
        except Exception as err:
            print("Unexpected image - ", imagesPathArray[i], " skipping...", err)
            
    return dataset, labels


def get_next_batch(data_path):

    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
               
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    
    # Reshape image data into the original shape
    image = tf.reshape(image, [input_width, input_width, input_depth])

    # Any preprocessing here ...

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=30, num_threads=1, min_after_dequeue=10)

    return images, labels
    
    

    

    
    
    
    
