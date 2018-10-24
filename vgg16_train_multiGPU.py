

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




dataset_folder = "D:/Dataset/Imagenet2012/"
training_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_train/"
testing_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_test/"
validation_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_val/"
saved_model_filepath = "./model/vgg16.ckpt"    # File path for saving the trained model


converted_train_data_filepath = "./data/imagenet_train_data.tfrecords"
converted_test_data_filepath = "./data/imagenet_test_data.tfrecords"
converted_val_data_filepath = "./data/imagenet_val_data.tfrecords"


train_label_file = './data/labels.txt'
test_label_file = './data/ILSVRC2012_test_ground_truth.txt'
validation_label_file = './data/ILSVRC2012_validation_ground_truth.txt'


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



training_image_folders = os.listdir(training_dataset_folder)
for each in training_image_folders:
    if ".tar" in each:
        training_image_folders.remove(each)
print("Training folders : ", len(training_image_folders))
    
    
    
def getListOfImages(training_image_folders):
    #start = time.time()
    
    all_training_images = []
    all_training_image_labels = []
    
    for f in range(len(training_image_folders)):
        images = os.listdir(training_dataset_folder + training_image_folders[f])
        for im in range(len(images)):
            image_path = training_dataset_folder + training_image_folders[f] + "/" + images[im]
            all_training_images.append(image_path)
            all_training_image_labels.append(f)
            
    #end = time.time()
    #print("Getting list of images, ", end-start)
    
    return all_training_images, all_training_image_labels

    
        
# Return the batch of training data for next run
def get_next_batch_of_training_images(imagesPathArray, image_labels):

    #start = time.time()
    
    dataset = np.ndarray(shape=(0, input_width, input_width, input_depth), dtype=np.float32)
    labels = np.ndarray(shape=(0, output_classes), dtype=np.float32)

    for i in range(len(imagesPathArray)):

        try:
            
            imarray = dp.decode_image_opencv(imagesPathArray[i])
            
            imlabel = dp.get_label(image_labels[i]) 
            
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
                
            imarray = dp.decode_image_opencv(pathToImage)

            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array([getNumber(labels_array[i])], dtype=np.float32)
            
            dataset = np.append(dataset, appendingImageArray, axis=0)
            labels = np.append(labels, appendingNumberLabel, axis=0)
        
        except Exception as err:
            print("Unexpected image - ", imagesPathArray[i], " skipping...", err)
            
    return dataset, labels





# Initialize parameters
training_epochs = 10
batch_size = 64
input_width = 224
input_depth = 3
display_step = 100
output_classes = len(training_GT)


learning_rate = 0.001
VGG_MEAN = [123.68, 116.779, 103.939]
means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
weight_decay = 0.0005
num_gpu = 2



# Initialize input and output
x = tf.placeholder(tf.float32, shape=[None, input_width, input_width, input_depth])
y = tf.placeholder(tf.float32, shape=[None, output_classes])


# model
vgg = vgg16.vgg16(x)
logits = vgg.logits


# Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
#regularizer = tf.nn.l2_loss(vgg.fc1w) + tf.nn.l2_loss(vgg.fc2w)
#loss = tf.reduce_mean(loss + regularizer * weight_decay)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

prediction = vgg.probs
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# Define model saver
saver = tf.train.Saver()
    
    


if __name__ == '__main__':

    # Tensorboard
    accuracy_history = tf.placeholder(tf.float32)
    accuracy_history_summary = tf.summary.scalar('training_history', accuracy_history)
    merged_history = tf.summary.merge_all()

    
    all_training_images, all_training_image_labels = getListOfImages(training_image_folders)

    print("Number of total training images: ", len(all_training_images))
    
    
    
    # test and validation images
    val_image_names = os.listdir(validation_dataset_folder)

    
    
    # Launch graph/Initialize session 
    with tf.Session() as sess:
        
        #start = time.time()
        print("Initializing global variables...")
        
        sess.run(tf.global_variables_initializer())
        
        writer_train = tf.summary.FileWriter('./model/hist/train',sess.graph)
        writer_test = tf.summary.FileWriter('./model/hist/test',sess.graph)
        writer_loss = tf.summary.FileWriter('./model/hist/loss',sess.graph)
        
        #end = time.time()
        #print("Initializing global variables, ", end-start)
        
        # Starting time
        t1 = time.time()
        
        
        for epoch in range(1, training_epochs + 1):
            start_epoch = time.time()
            train_accuracy_list = []
            test_accuracy_list = []
            loss_list = []
            
            print("\n")
            print("Starting epoch: ", epoch)
            
            # Shuffle images
            all_training_images, all_training_image_labels = dp.shuffle_images_np(all_training_images, all_training_image_labels)
            
         
            start_index = 0
            num_images = len(all_training_images)
            num_steps = round(num_images / batch_size)
            
        
            for j in range(num_steps):
                start = time.time()
                try:
                    print("\nStarting Epoch:", epoch, ", batch:", j + 1)
               
                    last_index = start_index + batch_size
                
                    step_images = all_training_images[start_index:last_index]
                    step_labels = all_training_image_labels[start_index:last_index]
                    
                    batch_train_images, batch_train_labels = get_next_batch_of_training_images(step_images, step_labels)
                    
                    training_data = {x: batch_train_images, y: batch_train_labels}
                    
                    sess.run(optimizer, feed_dict=training_data)
                    
                    
                    # saving model, accuracy and loss
                    if (j + 1) % display_step == 0 and j > 0:
                    
                        train_accuracy = accuracy.eval(feed_dict=training_data)
                        loss_print = loss.eval(feed_dict=training_data)
                        print("Epoch:", epoch, ", batch:", j + 1, ", Loss:", loss_print, ", Training Accuracy:", train_accuracy)
                        
                        train_accuracy_list.append(train_accuracy)
                        loss_list.append(loss_print)

             
                except Exception as err:
                    print("Error in training! Epoch:", epoch, ", batch:", j + 1, err)            

                start_index = start_index + batch_size
                
                end = time.time()
                print("time needed for this batch:", end - start, "seconds")
                
            training_accuracy = np.mean(train_accuracy_list)
            loss_here = np.mean(loss_list)
            
            writer_train.add_summary(sess.run(merged_history, feed_dict={accuracy_history: training_accuracy}), epoch)
            writer_loss.add_summary(sess.run(merged_history, feed_dict={accuracy_history: loss_here}), epoch)
        
            
            savedPath = saver.save(sess, saved_model_filepath)
            print("Model saved after epoch ", epoch, " at: " , savedPath)
            
            
            # Run validation images
            val_acc_list = []
            start_index = 0
            num_images = len(val_image_names)
            num_steps = round(num_images / batch_size)
                
                
            for m in range(num_steps):            
                try:
                    last_index = start_index + batch_size
                    step_images = image_names[start_index:last_index]
                    step_labels = validation_GT[start_index:last_index]

                    
                    batch_val_images, batch_val_labels = get_testing_images(step_images, step_labels, "validation")
                    val_data = {x: batch_val_images, y: batch_val_labels}
                    
                    
                    # logging
                    val_acc = accuracy.eval(feed_dict=val_data)
                    val_acc_list.append(val_acc)
                    
                except Exception as err:
                    print("Error in validation step:", m, err)
                    
                start_index = start_index + batch_size

            validation_accuracy = np.mean(val_acc_list)
                
            writer_test.add_summary(sess.run(merged_history, feed_dict={accuracy_history: validation_accuracy}), epoch)
            
            print("Training accuracy: ", training_accuracy * 100, ", Loss: ", loss_here, "%, Validation accuracy: ", testing_accuracy * 100, "%, ", epoch, "th epoch.")

            end_epoch = time.time()
            print("Total epoch time, ", end_epoch-start_epoch)
            
                
        print("\n")
        # Ending time
        t2 = time.time()
                
        # Save trained model
        savedPath = saver.save(sess, saved_model_filepath)
        print("Final model saved at: " , savedPath)
        
            
        print("Learning time: " + str(t2-t1) + " seconds")
        

        
print("\nTraining finished!")






