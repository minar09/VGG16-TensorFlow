

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
saved_model_filepath = "./model/vgg16.ckpt"    # File path for saving the trained model


# Initialize parameters

training_epochs = 10
batch_size = 64
image_width = 224
output_classes = 1000

learning_rate = 0.001

training_dataset_size = len(os.listdir(training_dataset_folder))
testing_dataset_size = len(os.listdir(testing_dataset_folder))
    
        
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


# Tensorboard
accuracy_history = tf.placeholder(tf.float32)
accuracy_history_summary = tf.summary.scalar('training_history', accuracy_history)
merged_history = tf.summary.merge_all()


if __name__ == '__main__':        
    
    # Define model saver
    saver = tf.train.Saver()

    # Launch graph/Initialize session 
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        
        writer_train = tf.summary.FileWriter('./model/hist/train',sess.graph)
        writer_test = tf.summary.FileWriter('./model/hist/test',sess.graph)
        writer_loss = tf.summary.FileWriter('./model/hist/loss',sess.graph)
        
        # Starting time
        t1 = time.time()
        
        training_steps = round(training_dataset_size / batch_size)
        testing_steps = round(testing_dataset_size / batch_size)
        
        
        for epoch in range(1, training_epochs + 1):
        
            train_accuracy_list = []
            test_accuracy_list = []
            loss_list = []
        
            for step in range(1, training_steps + 1):
                try:
                    batch_train_images, batch_train_labels = getBatchOfGTImages(batch_size, "training")
                    training_data = {x: batch_train_images, y: batch_train_labels}
                    sess.run(optimizer, feed_dict=training_data)
                    
                    # logging
                    train_accuracy = accuracy.eval(feed_dict=training_data)
                    loss_print = loss.eval(feed_dict=training_data)
                    train_accuracy_list.append(train_accuracy)
                    loss_list.append(loss_print)
                    
                except:
                    print("Something is wrong in training!")
                    
                    
            training_accuracy = np.mean(train_accuracy_list)
            loss_here = np.mean(loss_list)
            
            writer_train.add_summary(sess.run(merged_history, feed_dict={accuracy_history: training_accuracy}),e)
            writer_loss.add_summary(sess.run(merged_history, feed_dict={accuracy_history: loss_here}),e)
                    
            for step in range(1, testing_steps + 1):
                try:
                    batch_test_images, batch_test_labels = getBatchOfGTImages(batch_size, "testing")
                    testing_data = {x: batch_test_images, y: batch_test_labels}
                    
                    # logging
                    test_accuracy = accuracy.eval(feed_dict=testing_data)
                    test_accuracy_list.append(test_accuracy)
                except:
                    print("Something is wrong in testing!")
            
            testing_accuracy = np.mean(test_accuracy_list)
            writer_test.add_summary(sess.run(merged_history, feed_dict={accuracy_history: testing_accuracy}),e)
            
            print("Training accuracy: ", training_accuracy * 100, ", Loss: ", loss, "%, Testing accuracy: ", testing_accuracy * 100, "%, ", epoch, "th epoch.")
            
                
        # Ending time
        t2 = time.time()
                
        # Save trained model
        savedPath = saver.save(sess, saved_model_filepath)
        print("Model saved at: " ,savedPath)
        
            
    print("Learning time: " + str(t2-t1) + " seconds")






