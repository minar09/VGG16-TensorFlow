

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
validation_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_val/"
saved_model_filepath = "./model/vgg16.ckpt"    # File path for saving the trained model


test_label_file = 'ILSVRC2012_test_ground_truth.txt'
validation_label_file = 'ILSVRC2012_validation_ground_truth.txt'

f = open(test_label_file, 'r')
testing_GT = f.readlines()
f.close()

f = open(validation_label_file, 'r')
validation_GT = f.readlines()
f.close()


# Initialize parameters

training_epochs = 10
batch_size = 128
image_width = 224
output_classes = 1000

learning_rate = 0.001

training_dataset_size = len(os.listdir(training_dataset_folder))
testing_dataset_size = len(os.listdir(testing_dataset_folder))

class_folders = os.listdir(training_dataset_folder)

              
# Return the batch of training data for next run
def get_next_batch_of_training_images(class_folder, imagesPathArray, class_number):
     
    dataset = np.ndarray(shape=(0, image_width, image_width, 3), dtype=np.float32)
    labels = np.ndarray(shape=(0, output_classes), dtype=np.float32)
    
    for i in range(len(imagesPathArray)):

        try:
            pathToImage = training_dataset_folder + class_folder + imagesPathArray[i]
            image = imread(str(pathToImage), mode='RGB')
            imarray = imresize(image, (image_width, image_width))
            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array([class_number], dtype=np.float32)
            dataset = np.append(dataset, appendingImageArray, axis=0)
            labels = np.append(labels, appendingNumberLabel, axis=0)
        except:
            print("Unexpected image - ", imagesPathArray[i], ", skipping...")
            
    return dataset, labels
            
            
            
def get_testing_images(dataset_folder, imagesPathArray, labels_array):

    dataset = np.ndarray(shape=(0, image_width, image_width, 3), dtype=np.float32)
    labels = np.ndarray(shape=(0, output_classes), dtype=np.float32)
    
    for i in range(len(imagesPathArray)):

        try:
            pathToImage = testing_dataset_folder + imagesPathArray[i]
            image = imread(str(pathToImage), mode='RGB')
            imarray = imresize(image, (image_width, image_width))
            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array([labels_array[i]], dtype=np.float32)
            dataset = np.append(dataset, appendingImageArray, axis=0)
            labels = np.append(labels, appendingNumberLabel, axis=0)
        except:
            print("Unexpected image - ", imagesPathArray[i], ", skipping...")
            
    return dataset, labels
    

            
# Initialize input and output
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
            loss_list = []
            
            print("Starting epoch: ", epoch)
            
        
            for i in range(training_dataset_size):
                try:
                    image_names = os.listdir(training_dataset_folder + class_folder[i])
                    batch_train_images, batch_train_labels = get_next_batch_of_training_images(class_folder[i], image_names, int(class_number[i])+1)
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
            
            writer_train.add_summary(sess.run(merged_history, feed_dict={accuracy_history: training_accuracy}), epoch)
            writer_loss.add_summary(sess.run(merged_history, feed_dict={accuracy_history: loss_here}), epoch)
                    

            try:
                image_names = os.listdir(testing_dataset_folder)
                batch_test_images, batch_test_labels = get_testing_images(testing_dataset_folder, image_names, testing_GT)
                testing_data = {x: batch_test_images, y: batch_test_labels}
                
                # logging
                test_accuracy = accuracy.eval(feed_dict=testing_data)
                writer_test.add_summary(sess.run(merged_history, feed_dict={accuracy_history: testing_accuracy}), epoch)
            except:
                print("Something is wrong in testing!")
            
            
            print("Training accuracy: ", training_accuracy * 100, ", Loss: ", loss_here, "%, Testing accuracy: ", testing_accuracy * 100, "%, ", epoch, "th epoch.")
            
                
        # Ending time
        t2 = time.time()
                
        # Save trained model
        savedPath = saver.save(sess, saved_model_filepath)
        print("Model saved at: " ,savedPath)
        
            
    print("Learning time: " + str(t2-t1) + " seconds")
    
    
    try:
        val_image_names = os.listdir(validation_dataset_folder)
        batch_test_images, batch_test_labels = get_testing_images(validation_dataset_folder, val_image_names, validation_GT)
        val_data = {x: batch_test_images, y: batch_test_labels}
        
        # logging
        validation_accuracy = accuracy.eval(feed_dict=val_data)
        
        print("Validation accuracy: ", validation_accuracy * 100)
    except:
        print("Something is wrong in testing!")


    



