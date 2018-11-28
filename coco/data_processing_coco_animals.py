
# @TODO Clean up the code !!!
# rename with _ for internal functions

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Hide the warning messages about CPU/GPU

import os.path, time
import tensorflow as tf
import numpy as np
from random import randint
from scipy.misc import imread, imresize

dataset_folder = "D:/Dataset/coco-animals/"
training_dataset_folder = "D:/Dataset/coco-animals/train/"
validation_dataset_folder = "D:/Dataset/coco-animals/val/"
saved_model_filepath = "./model/coco/vgg16-coco.ckpt"    # File path for saving the trained model

# Initialize parameters
training_epochs = 10
batch_size = 32
input_width = 224
input_depth = 3
display_step = 10
output_classes = 8

VGG_MEAN = [123.68, 116.779, 103.939]
means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
weight_decay = 0.0005


# Get label number from 0 to 999 for the given label name
def get_number(label):
    for i, name in range(class_names):
        if name == label or label in name:
            return i

            
# Convert number/integer label to one-hot matrix
def get_label(label):
    return np.eye(output_classes, dtype=np.float32)[int(label)]
    

# Read and resize image using openCV/scipy
def decode_image_opencv(pathToImage):
    try:
        image = imread(str(pathToImage), mode='RGB')
        imarray = imresize(image, (input_width, input_width))
        imarray = imarray.reshape(input_width, input_width, input_depth)
        
        return imarray
    except Exception as err:
        print("Exception: ", err)
        return None
    
    
# Read and resize image using TensorFlow
def decode_image_with_tf(pathToImage):
    imageContents = tf.read_file(str(pathToImage))
    image = tf.image.decode_jpeg(imageContents, channels=3)
    resized_image = tf.image.resize_images(image, [input_width, input_width])
    imarray = resized_image.eval()
    imarray = imarray.reshape(input_width, input_width, input_depth)
    
    return imarray


# Read and parse (resize, rescale) image using TensorFlow
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
    
    
# Augmentation for image (crop, flip)
def training_preprocess(image):
    crop_image = tf.random_crop(image, [input_width, input_width, input_depth])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)          
    centered_image = flip_image - means                               

    return centered_image
    
    
# Augmentation for input (crop, centered)
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, input_width, input_width)  

    centered_image = crop_image - means                                    

    return centered_image, label
    
    
# Shuffle list of inputs and labels using numpy vectorization
def shuffle_images_np(imagesPathArray, imagesLabelsArray):
    start = time.time()
    print("Shuffling the images...")
    
    dataset_size = len(imagesPathArray)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    imagesPathArray = np.array(imagesPathArray)[indices.astype(int)]
    imagesPathArray = imagesPathArray[indices]
    
    imagesLabelsArray = np.array(imagesLabelsArray)[indices.astype(int)]
    imagesLabelsArray = imagesLabelsArray[indices]
    
    print("Time for shuffling data:", "{0:.2f}".format(time.time()-start), "sec")
    return imagesPathArray, imagesLabelsArray
    
    
# Shuffle list of inputs and labels using numpy randint
def shuffle_images_rand(imagesPathArray, imagesLabelsArray):
    
    print("Shuffling the images...")
    
    for i in range(0, len(imagesPathArray)):
        randomIndex1 = randint(0, len(imagesPathArray)-1)
        randomIndex2 = randint(0, len(imagesPathArray)-1)
        imagesPathArray[randomIndex1], imagesPathArray[randomIndex2] = imagesPathArray[randomIndex2], imagesPathArray[randomIndex1]
        imagesLabelsArray[randomIndex1], imagesLabelsArray[randomIndex2] = imagesLabelsArray[randomIndex2], imagesLabelsArray[randomIndex1]
    
    return imagesPathArray, imagesLabelsArray

    
# Get list of image paths and labels for ImageNet2012 training dataset
def get_list_of_training_data(training_image_folders):
    
    all_training_images = []
    all_training_image_labels = []
    
    for f in range(len(training_image_folders)):
        images = os.listdir(training_dataset_folder + training_image_folders[f])
        for im in range(len(images)):
            image_path = training_dataset_folder + training_image_folders[f] + "/" + images[im]
            #print(f, image_path)
            all_training_images.append(image_path)
            all_training_image_labels.append(f)
            
    return all_training_images, all_training_image_labels

    
# Get training data and labels for ImageNet2012 dataset
def _get_training_images():
    # Start time for listing training images
    start = time.time()
    training_image_folders = os.listdir(training_dataset_folder)
    for each in training_image_folders:
        if ".tar" in each:
            training_image_folders.remove(each)
    print("Training folders : ", len(training_image_folders))

    all_training_images, all_training_image_labels = get_list_of_training_data(training_image_folders)

    print("Number of total training images: ", len(all_training_images))
    print("Time for gathering training data:", "{0:.2f}".format(time.time()-start), "sec")
        
    return all_training_images, all_training_image_labels
    

# Return the batch of training data for next run
def get_next_batch_of_training_images(imagesPathArray, image_labels):
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

    return dataset, labels
            
            
# get file list and labels for ImageNet2012 test and validation data           
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
            appendingNumberLabel = np.array([get_label(labels_array[i])], dtype=np.float32)
            
            dataset = np.append(dataset, appendingImageArray, axis=0)
            labels = np.append(labels, appendingNumberLabel, axis=0)
        
        except Exception as err:
            print("Unexpected image - ", imagesPathArray[i], " skipping...", err)
            
    return dataset, labels

    
""" 
 preprocessing 
 load  a jpeg image and resize and cast 
 make one-hot encoding  for labels 
"""   
def _parse_jpeg(filename, label):

    # it called once, not repeatedly. When it make a graph 
    #print("filename:", filename, " label:", label)
    # you can debug how tensors flow in runtime
    #filename = tf.Print(filename, [filename]
    
    try:
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [input_width, input_width])
        #
        # ... More preprocessing ...
        crop_image = tf.random_crop(image_resized, [input_width, input_width, input_depth])
        flip_image = tf.image.random_flip_left_right(crop_image)
        #
        image = tf.cast(flip_image, tf.float32)
        #image = tf.cast(image_resized, tf.float32)
        
        #label = tf.cast(label, tf.float32)
        label = tf.one_hot(label, output_classes)
        print(image.shape, label.shape)
        return image, label

    except Exception as err:
        print(err)
        
    
"""

    external functions 

"""

"""
  create Dataset of list of files and labels (not images)
  Due to the huge size of IMAGENET dataset, the loading is done in parser 
  
  training, validation, testing
  
"""    
    
def create_raw_dataset():
    
    # 1. Setup the file list and labels in list format    
    # A vector of filenames.
    print("Generating the dataset...")

    images_path_list = []
    images_label_list = []        
    image_names, image_labels = _get_images()
    
    filenames = tf.constant(image_names)
    labels = tf.constant(image_labels)     
    print("Raw dataset:", filenames.shape, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    return dataset
    

#
# add preprocessing option to the dataset 
#         
def add_pipeline(dataset, batchsz, num, nthread): 

    print("Adding pipelining and preprocessing...")   

    try:
        dataset = dataset.map(_parse_jpeg, num_parallel_calls = nthread)
        #dataset = dataset.repeat(training_epochs)
        dataset = dataset.shuffle(buffer_size=num)
        dataset = dataset.batch(batchsz)     
        dataset = dataset.prefetch(batchsz)  # preload dataset (image decoding) 
        return dataset
    except Exception as err:
        print(err)
        

# Get data and labels for coco-animals dataset
def _get_images():
    # Start time for listing images
    start = time.time()

    training_image_folders = os.listdir(training_dataset_folder)

    all_training_images = []
    all_training_image_labels = []
    
    for f in range(len(training_image_folders)):
    
        # training folder
        images = os.listdir(training_dataset_folder + training_image_folders[f])
        for im in range(len(images)):
            image_path = training_dataset_folder + training_image_folders[f] + "/" + images[im]
            #print(f, image_path)
            all_training_images.append(image_path)
            all_training_image_labels.append(f)
            
        # validation folder
        images = os.listdir(validation_dataset_folder + training_image_folders[f])
        for im in range(len(images)):
            image_path = validation_dataset_folder + training_image_folders[f] + "/" + images[im]
            #print(f, image_path)
            all_training_images.append(image_path)
            all_training_image_labels.append(f)

    print("Number of total images: ", len(all_training_images))
        
    return all_training_images, all_training_image_labels
        
    print("Time for gathering data:", "{0:.2f}".format(time.time()-start), "sec")

    