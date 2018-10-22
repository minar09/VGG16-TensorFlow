

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import vgg16
import os, sys
import data_processing as dp



dataset_folder = "D:/Dataset/Imagenet2012/"
training_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_train/"
testing_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_test/"
validation_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_val/"
saved_model_filepath = "./model/vgg16.ckpt"    # File path for saving the trained model


train_label_file = 'labels.txt'
test_label_file = 'ILSVRC2012_test_ground_truth.txt'
validation_label_file = 'ILSVRC2012_validation_ground_truth.txt'


f = open(train_label_file, 'r')
training_GT = f.readlines()
f.close()

print("Number of training labels: ", len(training_GT))


# Initialize parameters

training_epochs = 50
batch_size = 64
input_width = 224
input_depth = 3
output_classes = 1000

learning_rate = 0.001
VGG_MEAN = [123.68, 116.779, 103.939]
means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])


def getNumber(label):
    for i, name in range(class_names):
        if name == label or label in name:
            return i
            
            
def decode_image(pathToImage):
    image = imread(str(pathToImage), mode='RGB')
    imarray = imresize(image, (input_width, input_width))
    #print("\nStarting reading image - ", pathToImage)
    #imageContents = tf.read_file(str(pathToImage))
    #print("Starting decoding image")
    #image = tf.image.decode_jpeg(imageContents, channels=3)
    #print("Starting resizing image")
    #resized_image = tf.image.resize_images(image, [input_width, input_width])
    #imarray = resized_image.eval()
    #print("Starting reshaping image")
    imarray = imarray.reshape(input_width, input_width, input_depth)
    
    return imarray
            

    
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
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

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image.eval()
    
    
def training_preprocess(image):
    crop_image = tf.random_crop(image, [input_width, input_width, input_depth])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

    centered_image = flip_image - means                                     # (5)

    return centered_image
    
    
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, input_width, input_width)    # (3)

    centered_image = crop_image - means                                     # (4)

    return centered_image, label
    
    

x = tf.placeholder(tf.float32, shape=[None, input_width, input_width, input_depth])
y = tf.placeholder(tf.float32, shape=[None, output_classes])


# model
vgg = vgg16.vgg16(x)
logits = vgg.logits


# Loss function using L2 Regularization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

prediction = vgg.probs
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def run_testing_images(sess):
    
    start_index = 0
    num_images = len(test_image_names)
    num_steps = round(num_images/batch_size)
    

    for k in range(num_steps):            
        try:
            last_index = start_index + batch_size
            step_images = image_names[start_index:last_index]
            step_labels = testing_GT[start_index:last_index]
            
            batch_test_images, batch_test_labels = get_testing_images(step_images, step_labels, "testing")
            testing_data = {x: batch_test_images, y: batch_test_labels}
            
            # logging
            test_accuracy = accuracy.eval(feed_dict=testing_data)
            test_accuracy_list.append(test_accuracy)
            
        except Exception as err:
            print("Error in testing step:", k, err)
            
        start_index = start_index + batch_size
    
    
    testing_accuracy = np.mean(test_accuracy_list)
    print("Testing accuracy: ", testing_accuracy * 100)

	
	
	

dataset_folder = "D:/Dataset/Imagenet2012/"
training_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_train/"
testing_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_test/"
validation_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_val/"
saved_model_filepath = "./model/vgg16.ckpt"    # File path for saving the trained model


train_label_file = 'labels.txt'
test_label_file = 'ILSVRC2012_test_ground_truth.txt'
validation_label_file = 'ILSVRC2012_validation_ground_truth.txt'


training_GT = []
testing_GT = []
validation_GT = []



f = open(train_label_file, 'r')
training_GT = f.readlines()
f.close()

print("Number of training labels: ", len(training_GT))

f = open(test_label_file, 'r')
testing_GT = f.readlines()
f.close()

print("Number of testing images: ", len(testing_GT))

f = open(validation_label_file, 'r')
validation_GT = f.readlines()
f.close()

print("Number of validation images: ", len(validation_GT))



testing_dataset_size = len(os.listdir(testing_dataset_folder))
validation_dataset_size = len(os.listdir(validation_dataset_folder))




# Initialize parameters
training_epochs = 50
batch_size = 64
input_width = 224
input_depth = 3
display_step = 100
output_classes = len(training_GT)


learning_rate = 0.001
VGG_MEAN = [123.68, 116.779, 103.939]
means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])


# define the model

# Initialize input and output
x = tf.placeholder(tf.float32, shape=[None, input_width, input_width, input_depth])
y = tf.placeholder(tf.float32, shape=[None, output_classes])


# model
vgg = vgg16.vgg16(x)
logits = vgg.logits


# Loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = dp.learning_rate).minimize(cost)

prediction = vgg.probs
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# test and validation images
test_image_names = os.listdir(dp.testing_dataset_folder)




        
def main(argv):
    
    with tf.Session() as sess:
    
        if(argv):
        
            argv = argv[1]
    
            imgs = tf.placeholder(tf.float32, [None, input_width, input_width, input_depth])
            vgg = vgg16(imgs, "./model/vgg16.ckpt", sess)

            img1 = decode_image(argv)
            #image = _parse_function(argv)
            #img1 = val_preprocess(image)

            prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
            preds = (np.argsort(prob)[::-1])[0:5]
            for p in preds:
                print(class_names[p], prob[p])
                
        else:
            run_testing_images(sess)
    
    
if __name__ == "__main__":
    main(sys.argv)
            
            
            
            