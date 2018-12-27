
# Hide the warning messages about CPU/GPU
import os.path
import data_processing as dp
import vgg16
from imagenet_classes import class_names
from scipy.misc import imread, imresize
from random import randint
import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


dataset_folder = "D:/Dataset/Imagenet2012/"
training_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_train/"
testing_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_test/"
validation_dataset_folder = "D:/Dataset/Imagenet2012/Images/ILSVRC2012_img_val/"
# File path for saving the trained model
saved_model_filepath = "./model/vgg16.ckpt"

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
VGG_MEAN = [123.68, 116.779, 103.939]
means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
weight_decay = 0.0005

# Initialize input and output
x = tf.placeholder(tf.float32, shape=[
                   None, input_width, input_width, input_depth])
y = tf.placeholder(tf.float32, shape=[None, output_classes])

# model
vgg = vgg16.vgg16(x)
logits = vgg.logits

# Loss function
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
#regularizer = tf.nn.l2_loss(vgg.fc1w) + tf.nn.l2_loss(vgg.fc2w)
#loss = tf.reduce_mean(loss + regularizer * weight_decay)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = vgg.probs
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define model saver
saver = tf.train.Saver()

if __name__ == '__main__':

    # Tensorboard
    accuracy_history = tf.placeholder(tf.float32)
    accuracy_history_summary = tf.summary.scalar(
        'training_history', accuracy_history)
    merged_history = tf.summary.merge_all()

    # training images
    all_training_images, all_training_image_labels = dp.get_training_images()

    # test and validation images
    val_image_names = os.listdir(validation_dataset_folder)

    # Launch graph/Initialize session
    with tf.Session() as sess:

        #start = time.time()
        print("Initializing global variables...")

        sess.run(tf.global_variables_initializer())

        writer_train = tf.summary.FileWriter('./model/hist/train', sess.graph)
        writer_test = tf.summary.FileWriter('./model/hist/test', sess.graph)
        writer_loss = tf.summary.FileWriter('./model/hist/loss', sess.graph)

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
            all_training_images, all_training_image_labels = dp.shuffle_images_np(
                all_training_images, all_training_image_labels)

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

                    batch_train_images, batch_train_labels = dp.get_next_batch_of_training_images(
                        step_images, step_labels)

                    training_data = {
                        x: batch_train_images, y: batch_train_labels}

                    sess.run(optimizer, feed_dict=training_data)

                    # saving model, accuracy and loss
                    if (j + 1) % display_step == 0 and j > 0:

                        train_accuracy = accuracy.eval(feed_dict=training_data)
                        loss_print = loss.eval(feed_dict=training_data)
                        print("Epoch:", epoch, ", batch:", j + 1, ", Loss:",
                              loss_print, ", Training Accuracy:", train_accuracy)

                        train_accuracy_list.append(train_accuracy)
                        loss_list.append(loss_print)

                except Exception as err:
                    print("Error in training! Epoch:",
                          epoch, ", batch:", j + 1, err)

                start_index = start_index + batch_size

                end = time.time()
                print("time needed for this batch:", end - start, "seconds")

            training_accuracy = np.mean(train_accuracy_list)
            loss_here = np.mean(loss_list)

            writer_train.add_summary(sess.run(merged_history, feed_dict={
                                     accuracy_history: training_accuracy}), epoch)
            writer_loss.add_summary(sess.run(merged_history, feed_dict={
                                    accuracy_history: loss_here}), epoch)

            savedPath = saver.save(sess, saved_model_filepath)
            print("Model saved after epoch ", epoch, " at: ", savedPath)

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

                    batch_val_images, batch_val_labels = dp.get_testing_images(
                        step_images, step_labels, "validation")
                    val_data = {x: batch_val_images, y: batch_val_labels}

                    # logging
                    val_acc = accuracy.eval(feed_dict=val_data)
                    val_acc_list.append(val_acc)

                except Exception as err:
                    print("Error in validation step:", m, err)

                start_index = start_index + batch_size

            validation_accuracy = np.mean(val_acc_list)

            writer_test.add_summary(sess.run(merged_history, feed_dict={
                                    accuracy_history: validation_accuracy}), epoch)

            print("Training accuracy: ", training_accuracy * 100, ", Loss: ", loss_here,
                  "%, Validation accuracy: ", testing_accuracy * 100, "%, ", epoch, "th epoch.")

            end_epoch = time.time()
            print("Total epoch time, ", end_epoch-start_epoch)

        print("\n")
        # Ending time
        t2 = time.time()

        # Save trained model
        savedPath = saver.save(sess, saved_model_filepath)
        print("Final model saved at: ", savedPath)

        print("Learning time: " + str(t2-t1) + " seconds")

print("\nTraining finished!")
