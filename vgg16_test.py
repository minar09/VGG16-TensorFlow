

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import vgg16
import os, sys




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

training_epochs = 30
batch_size = 32
image_width = 224
output_classes = 1000

training_data_percentage = 0.8
learning_rate = 0.001


def getNumber(label):
    for i, name in range(class_names):
        if name == label or label in name:
            return i
            
            
def decode_image(pathToImage):
    #image = imread(str(pathToImage), mode='RGB')
    #imarray = imresize(image, (image_width, image_width))
    imageContents = tf.read_file(str(pathToImage))
    image = tf.image.decode_jpeg(imageContents, channels=3, dtype=tf.uint8)
    resized_image = tf.image.resize_images(image, [image_width, image_width, 3])
    imarray = resized_image.eval()
    imarray = imarray.reshape(image_width, image_width, 3)
    
    return imarray
            
            

x = tf.placeholder(tf.float32, shape=[None, image_width, image_width, 3])
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


        
def main(argv):
    
    with tf.Session() as sess:
    
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg16(imgs, "./model/vgg16.ckpt", sess)

        img1 = decode_image(argv)

        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(class_names[p], prob[p])
    
    
if __name__ == "__main__":
    main(sys.argv[1])
            
            