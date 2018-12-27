
# @TODO Clean up the code !!!
# rename with _ for internal functions

import time
import os.path
from imagenet_classes import class_names
from scipy.misc import imread, imresize
from random import randint
import numpy as np
import tensorflow as tf
import os
# Hide the warning messages about CPU/GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


dataset_folder = "D:/Dataset/Imagenet2012/"
training_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_train/"
testing_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_test/"
validation_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_val/"

test_label_file = './data/ILSVRC2012_test_ground_truth.txt'
validation_label_file = './data/ILSVRC2012_validation_ground_truth.txt'
train_selected_file = './data/train_selected.txt'

training_GT = []
testing_GT = []
validation_GT = []
selected_classes = []

#training_dataset_size = 1281167
training_dataset_size = 103732
testing_dataset_size = len(os.listdir(testing_dataset_folder))
validation_dataset_size = len(os.listdir(validation_dataset_folder))

# Get labels/GT for test data
f = open(test_label_file, 'r')
testing_GT = f.readlines()
f.close()
#print("Number of testing images: ", len(testing_GT))

# Get labels/GT for validation data
f = open(validation_label_file, 'r')
validation_GT = f.readlines()
# print(validation_GT)
f.close()
#print("Number of validation images: ", len(validation_GT))

# Get selected classes
selected_classes = [1, 2, 8, 13, 21, 24, 32, 37, 49, 63, 71, 79, 84, 99, 109, 122, 130, 145, 184, 277, 284, 292, 306, 314, 319, 328, 331, 340, 356, 366, 388, 397, 402, 407, 409, 412, 417, 430, 440, 448,
                    457, 468, 480, 498, 515, 539, 563, 579, 587, 604, 610, 614, 620, 624, 632, 637, 640, 651, 655, 668, 673, 691, 707, 713, 723, 736, 764, 812, 852, 861, 879, 883, 916, 928, 937, 947, 954, 963, 980, 999]
print("Number of selected classes: ", len(selected_classes))

# Initialize parameters
training_epochs = 50
input_width = 224
input_depth = 3
#output_classes = 1000
output_classes = 80

VGG_MEAN = [123.68, 116.779, 103.939]
means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])


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
    crop_image = tf.random_crop(
        image, [input_width, input_width, input_depth])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)
    centered_image = flip_image - means

    return centered_image


# Augmentation for input (crop, centered)
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(
        image, input_width, input_width)

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

    print("Time for shuffling data:",
          "{0:.2f}".format(time.time()-start), "sec")
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

    # for f in range(len(training_image_folders)):
    # for f in range(5):
    label = 0    # class label for selected classes, otherwise folder iteration number
    for f in selected_classes:
        # print(f)
        images = os.listdir(training_dataset_folder +
                            training_image_folders[f])
        for im in range(len(images)):
            # for im in range(10):
            image_path = training_dataset_folder + \
                training_image_folders[f] + "/" + images[im]
            all_training_images.append(image_path)
            # all_training_image_labels.append(f)
            all_training_image_labels.append(label)
            #print(label, image_path)

        label = label + 1    # label increment for selected classes

    return all_training_images, all_training_image_labels


# Get training data and labels for ImageNet2012 dataset
def _get_training_images():
    # Start time for listing training images
    start = time.time()
    training_image_folders = os.listdir(training_dataset_folder)

    for each in training_image_folders:
        if ".tar" in each:
            training_image_folders.remove(each)

    #print("Training folders : ", len(training_image_folders))

    all_training_images, all_training_image_labels = get_list_of_training_data(
        training_image_folders)

    training_dataset_size = len(all_training_images)

    print("Number of total training images: ", training_dataset_size)
    print("Time for gathering training data:",
          "{0:.2f}".format(time.time()-start), "sec")

    return all_training_images, all_training_image_labels


# Return the batch of training data for next run
def get_next_batch_of_training_images(imagesPathArray, image_labels):
    dataset = np.ndarray(
        shape=(0, input_width, input_width, input_depth), dtype=np.float32)
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
            print("Unexpected image - ",
                  imagesPathArray[i], ", skipping...", image_labels[i], err)

    return dataset, labels


# get file list and labels for ImageNet2012 test and validation data
def get_testing_images(imagesPathArray, labels_array, type="testing"):

    dataset = np.ndarray(
        shape=(0, input_width, input_width, input_depth), dtype=np.float32)
    labels = np.ndarray(shape=(0, output_classes), dtype=np.float32)

    for i in range(len(imagesPathArray)):

        try:
            pathToImage = testing_dataset_folder + imagesPathArray[i]
            if type == "validation":
                pathToImage = validation_dataset_folder + imagesPathArray[i]

            imarray = decode_image_opencv(pathToImage)

            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array(
                [get_label(labels_array[i])], dtype=np.float32)

            dataset = np.append(dataset, appendingImageArray, axis=0)
            labels = np.append(labels, appendingNumberLabel, axis=0)

        except Exception as err:
            print("Unexpected image - ",
                  imagesPathArray[i], " skipping...", err)

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
    # filename = tf.Print(filename, [filename]

    try:
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        #image_resized = tf.image.resize_images(image_decoded, [input_width, input_width])
        image_resized_with_crop_pad = tf.image.resize_image_with_crop_or_pad(
            image_decoded, input_width, input_width)
        #
        # ... More preprocessing ...
        #
        image_flipped = tf.image.random_flip_left_right(
            image_resized_with_crop_pad)

        #image = tf.cast(image_resized, tf.float32)
        image = tf.cast(image_flipped, tf.float32)

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


def create_raw_dataset(data_type="training"):

    # 1. Setup the file list and labels in list format
    # A vector of filenames.
    print("Generating the dataset...")

    if data_type == "training":

        images_path_list = []
        images_label_list = []
        image_names, image_labels = _get_training_images()
        # do not need do it here, there is suffle method in dataset
        image_names, image_labels = shuffle_images_np(
            image_names, image_labels)

        filenames = tf.constant(image_names)
        labels = tf.constant(image_labels)
        print("Raw dataset:", filenames.shape, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        return dataset

    elif data_type == "validation":

        val_images_path_list = []
        val_images_label_list = []

        image_names = os.listdir(validation_dataset_folder)
        for i in range(len(image_names)):
            # for i in range(1000):
            fname = os.path.join(validation_dataset_folder + image_names[i])
            val_images_path_list.append(fname)
            val_images_label_list.append(int(validation_GT[i].split("\n")[0]))

        filenames = tf.constant(val_images_path_list)

        labels = tf.constant(val_images_label_list)
        print("Raw dataset:", filenames.shape, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        return dataset

    elif data_type == "testing":
        images_path_list = []
        images_label_list = []
        image_names = os.listdir(testing_dataset_folder)

        for i in range(len(image_names)):
            fname = os.path.join(testing_dataset_folder + image_names[i])
            images_path_list.append(fname)
            images_label_list.append(int(testing_GT[i].split("\n")[0]))

        filenames = tf.constant(images_path_list)

        #labels = tf.constant(testing_GT)
        labels = tf.constant(images_label_list)

        print("Raw dataset:", filenames.shape, labels.shape)

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        return dataset

#
# add preprocessing option to the dataset
#


def add_pipeline(dataset, batchsz, num, nthread):

    print("Adding pipelining and preprocessing...")

    try:
        dataset = dataset.map(_parse_jpeg, num_parallel_calls=nthread)
        #dataset = dataset.cache()
        #dataset = dataset.repeat(training_epochs)
        #dataset = dataset.shuffle(buffer_size=num)
        dataset = dataset.batch(batchsz)
        dataset = dataset.prefetch(batchsz)  # preload dataset (image decoding)

        return dataset
    except Exception as err:
        print(err)
