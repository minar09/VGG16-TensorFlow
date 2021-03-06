
# Hide the warning messages about CPU/GPU
import os.path
import data_processing_coco_animals as dp
import vgg16_coco_animals
from scipy.misc import imread, imresize
from random import randint
import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


dataset_folder = "D:/Dataset/coco-animals/"
training_dataset_folder = "D:/Dataset/coco-animals/train/"
validation_dataset_folder = "D:/Dataset/coco-animals/val/"
# File path for saving the trained model
saved_model_filepath = "./model/coco/vgg16-coco.ckpt"

# Initialize parameters
training_epochs = 5
batch_size = 32
input_width = 224
input_depth = 3
display_step = 10
output_classes = 8

learning_rate = 0.0001
VGG_MEAN = [123.68, 116.779, 103.939]
means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
weight_decay = 0.0005

# Initialize input and output
x = tf.placeholder(tf.float32, shape=[
                   None, input_width, input_width, input_depth])
y = tf.placeholder(tf.float32, shape=[None, output_classes])

# model
vgg = vgg16_coco_animals.vgg16(x)
logits = vgg.logits

# Loss function
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
vars = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * weight_decay
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=logits)) + lossL2
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
    all_training_images, all_training_image_labels = dp._get_images("training")

    # validation images
    all_validation_images, all_validation_image_labels = dp._get_images(
        "validation")

    # Launch graph/Initialize session
    with tf.Session() as sess:

        start = time.time()
        print("Initializing global variables...")

        sess.run(tf.global_variables_initializer())

        writer_train = tf.summary.FileWriter(
            './model/hist/coco/train', sess.graph)
        writer_test = tf.summary.FileWriter(
            './model/hist/coco/test', sess.graph)
        writer_loss = tf.summary.FileWriter(
            './model/hist/coco/loss', sess.graph)

        end = time.time()
        print("Time for initializing global variables is ", end-start, "secs")

        # Restore parameters from trained model
        ts = time.time()
        try:
            saver.restore(sess, saved_model_filepath)
        except Exception as err:
            print("Error in restoring model: ", err)

        print("Time for restoring params",
              "{0:.2f}".format(time.time()-ts), "sec")

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
                #start = time.time()
                try:
                    #print("\nStarting Epoch:", epoch, ", batch:", j + 1)

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

                #end = time.time()
                #print("time needed for this batch:", end - start, "seconds")

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
            num_images = len(all_validation_images)
            num_steps = round(num_images / batch_size)

            for m in range(num_steps):
                try:
                    last_index = start_index + batch_size

                    step_images = all_validation_images[start_index:last_index]
                    step_labels = all_validation_image_labels[start_index:last_index]

                    batch_val_images, batch_val_labels = dp.get_next_batch_of_training_images(
                        step_images, step_labels)
                    val_data = {x: batch_val_images, y: batch_val_labels}

                    # logging
                    val_acc = accuracy.eval(feed_dict=val_data)
                    val_acc_list.append(val_acc)

                    if (m + 1) % display_step == 0 and m > 0:
                        print("Epoch:", epoch, ", step:", m + 1,
                              ", Validation Accuracy:", val_acc)

                except Exception as err:
                    print("Error in validation step:", m, err)

                start_index = start_index + batch_size

            validation_accuracy = np.mean(val_acc_list)

            writer_test.add_summary(sess.run(merged_history, feed_dict={
                                    accuracy_history: validation_accuracy}), epoch)

            print("Training accuracy: ", training_accuracy * 100, "%, Loss: ", loss_here,
                  ", Validation accuracy: ", validation_accuracy * 100, "%, ", epoch, "th epoch.")

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
