

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


training_GT = []
testing_GT = []
validation_GT = []


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
training_epochs = 10
batch_size = 64
input_width = 224
input_depth = 3
display_step = 100
output_classes = 1000


learning_rate = 0.001
weight_decay = 0.0005



def parser(record):
    keys_to_features = {
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    #image = tf.image.decode_jpeg(parsed["image"])
    image = tf.decode_raw(parsed['image'], tf.float32)
    print(image.shape)
    image = tf.reshape(image, [input_width, input_width, input_depth])
    image = tf.cast(image, tf.float32)
    
    label = tf.cast(parsed["label"], tf.float32)
    #label = tf.one_hot(label, output_classes)
    
    print(image.shape, label.shape)

    return image, label


# Prepare dataset
filenames = [converted_test_data_filepath, converted_val_data_filepath]
dataset = tf.data.TFRecordDataset(filenames)

# Use `tf.parse_single_example()` to extract data from a `tf.Example`
# protocol buffer, and perform any additional per-record preprocessing.


# Use `Dataset.map()` to build a pair of a feature dictionary and a label
# tensor for each example.
dataset = dataset.map(parser)
dataset = dataset.cache()
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(training_epochs)

# Each element of `dataset` is tuple containing a dictionary of features
# (in which each value is a batch of values for that feature), and a batch of
# labels.


iterator = dataset.make_one_shot_iterator()
#iterator = dataset.make_initializable_iterator()
x, y = iterator.get_next()

#loss = model_function(next_example, next_label)

        


# Initialize input and output
#x = tf.placeholder(tf.float32, shape=[None, input_width, input_width, input_depth])
#y = tf.placeholder(tf.float32, shape=[None, output_classes])



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

    
    
    
"""feature = {'image': tf.FixedLenFeature([], tf.string),
       'label': tf.FixedLenFeature([], tf.int64)}
       
# Create a list of filenames and pass it to a queue
filename_queue = tf.train.string_input_producer([converted_test_data_filepath], num_epochs=training_epochs)

# Define a reader and read the next record
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

# Decode the record read by the reader
features = tf.parse_single_example(serialized_example, features=feature)

# Convert the image data from string back to the numbers
image = tf.decode_raw(features['image'], tf.float32)

# Cast label data into int32
label = tf.cast(features['label'], tf.int32)

# Reshape image data into the original shape
image = tf.reshape(image, [input_width, input_width, input_depth])

# one-hot encoding for label
label = tf.one_hot(label, output_classes)"""



    


if __name__ == '__main__':

    # Tensorboard
    accuracy_history = tf.placeholder(tf.float32)
    accuracy_history_summary = tf.summary.scalar('training_history', accuracy_history)
    merged_history = tf.summary.merge_all()

    
    
    # Launch graph/Initialize session 
    with tf.Session() as sess:
        
        
        print("Initializing global variables...")
        
        sess.run(tf.global_variables_initializer())
        
        writer_train = tf.summary.FileWriter('./model/hist/train',sess.graph)
        writer_test = tf.summary.FileWriter('./model/hist/test',sess.graph)
        writer_loss = tf.summary.FileWriter('./model/hist/loss',sess.graph)
        
        
        # Starting time
        t1 = time.time()
        
        
        for i in range(training_epochs):
            print("Starting Epoch", i)
            try:
                sess.run(optimizer)
            except Exception as err:
                print(err)
    
        """for epoch in range(1, training_epochs + 1):
            start_epoch = time.time()
            train_accuracy_list = []
            test_accuracy_list = []
            loss_list = []
            
            print("\n")
            print("Starting epoch: ", epoch)
   
   
            #num_images = 1281167
            num_images = 100000
            num_steps = num_images // batch_size
            
            
            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
        
            for j in range(num_steps):
                start = time.time()
                try:
                    print("\nStarting Epoch:", epoch, ", batch:", j + 1)
               
                    #batch_train_images, batch_train_labels = dp.get_next_batch(converted_test_data_filepath)
                    
                    # Creates batches by randomly shuffling tensors
                    batch_train_images, batch_train_labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=30, num_threads=1, min_after_dequeue=10)

                    #print(batch_train_images.shape, batch_train_labels.shape)
                    batch_train_images, batch_train_labels = sess.run([batch_train_images, batch_train_labels])
                    training_data = {x: batch_train_images, y: batch_train_labels}
                    
                    sess.run(optimizer, feed_dict=training_data)
                    #sess.run(optimizer, [batch_train_images, batch_train_labels])
                    #sess.run(batch_train_images, batch_train_labels)
                    
                    
                    # saving model, accuracy and loss
                    if (j + 1) % display_step == 0 and j > 0:
                    
                        train_accuracy = accuracy.eval(feed_dict=training_data)
                        loss_print = loss.eval(feed_dict=training_data)
                        print("Epoch:", epoch, ", batch:", j + 1, ", Loss:", loss_print, ", Training Accuracy:", train_accuracy)
                        
                        train_accuracy_list.append(train_accuracy)
                        loss_list.append(loss_print)

             
                except Exception as err:
                    print("Error in training! Epoch:", epoch, ", batch:", j + 1, err)            

                end = time.time()
                print("time needed for this batch:", end - start, "seconds")
                
            training_accuracy = np.mean(train_accuracy_list)
            loss_here = np.mean(loss_list)
            
            writer_train.add_summary(sess.run(merged_history, feed_dict={accuracy_history: training_accuracy}), epoch)
            writer_loss.add_summary(sess.run(merged_history, feed_dict={accuracy_history: loss_here}), epoch)
        
            
            savedPath = saver.save(sess, saved_model_filepath)
            print("Model saved after epoch ", epoch, " at: " , savedPath)
            
            
            # Stop the threads
            coord.request_stop()
            
            # Wait for threads to stop
            coord.join(threads)
            
            
            # Run validation images
            val_acc_list = []

            num_images = 50000
            num_steps = num_images // batch_size
                
                
            for m in range(num_steps):            
                try:
                    
                    batch_val_images, batch_val_labels = dp.get_next_batch(converted_val_data_filepath)
                    val_data = {x: batch_val_images, y: batch_val_labels}
                    
                    
                    # logging
                    val_acc = accuracy.eval(feed_dict=val_data)
                    val_acc_list.append(val_acc)
                    
                except Exception as err:
                    print("Error in validation step:", m, err)
                    
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
        
            
        print("Learning time: " + str(t2-t1) + " seconds")"""
        

        
print("\nTraining finished!")






