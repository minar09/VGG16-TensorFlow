

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import vgg16
import os, sys



# Initialize parameters

training_epochs = 30
batch_size = 128
image_width = 224
output_classes = 1000

training_data_percentage = 0.8
learning_rate = 0.001


def getNumber(label):
    for i, name in range(class_names):
        if name == label or label in name:
            return i
			
			

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

        img1 = imread(argv, mode='RGB')
        img1 = imresize(img1, (224, 224))

        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(class_names[p], prob[p])
	
    
if __name__ == "__main__":
    main(sys.argv[1])
			
			