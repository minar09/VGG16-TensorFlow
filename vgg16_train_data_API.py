# Hide the warning messages about CPU/GPU
from imagenet_classes import class_names    # imagenet
import data_processing as dp                # data input loading
import vgg                                  # VGG-16 model
import os.path
from scipy.misc import imread, imresize
from random import randint
import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# customer module

# model path
# File path for saving the trained model
saved_model_filepath = "./model/vgg16/vgg.ckpt"

#training_dataset_size = 1281167
training_dataset_size = 103732
testing_dataset_size = 100000
validation_dataset_size = 50000

# Initialize parameters
training_epochs = 50    # for testing
# due to the OOM after Regularization # 64     # limitation of GTX 1070 (8GB)
batch_size = 16
nThread = 16  # 24     # # of Dell server's CPU core
input_width = 224   # size of input for VGG model
input_depth = 3     # number of channels in the image
display_step = 100  # show
# output_classes = 1000   # labels/types/classes of input images
output_classes = 80   # labels/types/classes of input images

# learning_rate = 0.00001   # default learning rate
global_step = tf.Variable(0, trainable=False)
#starter_learning_rate = 0.01
starter_learning_rate = 0.00001
learning_rate = tf.train.exponential_decay(
    starter_learning_rate, global_step, 100000, 0.1, staircase=True)

start = time.time()   # Start time for initializing dataset and pipelining

# 1. input pipeline
rawds = dp.create_raw_dataset("training")
dataset = dp.add_pipeline(rawds, batch_size, training_dataset_size, nThread)
iterator = tf.data.Iterator.from_structure(
    dataset.output_types, dataset.output_shapes)
dataset_init_op = iterator.make_initializer(dataset)
#iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

print(images.shape, labels.shape)
print("Time for initializing dataset and pipelining:",
      "{0:.2f}".format(time.time()-start), "sec")

# Initialize input and output
x = images      # image batch
y = labels      # label batch

# 2. classifier model
# @TODO use stored weight and session  See  def __init__(self, imgs, weights=None, sess=None):
logits = vgg.build(x, n_classes=output_classes, training=True)
probs = tf.nn.softmax(logits)

# 3. Loss & optimizer function
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
vars = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.0005
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=logits)) + lossL2
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss)
#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(loss, global_step=global_step)

#prediction = vgg.probs
prediction = probs
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define model saver
saver = tf.train.Saver()


if __name__ == '__main__':

    # event logs for Tensorboard
    accuracy_history = tf.placeholder(tf.float32)
    accuracy_history_summary = tf.summary.scalar(
        'training_history', accuracy_history)
    merged_history = tf.summary.merge_all()

    # Launch graph/Initialize session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        global_start = time.time()
        print("Initializing global variables...")

        sess.run(tf.global_variables_initializer())

        # File writers for history
        writer_train = tf.summary.FileWriter('./hist/vgg16/train', sess.graph)
        writer_loss = tf.summary.FileWriter('./hist/vgg16/loss', sess.graph)
        writer_lr = tf.summary.FileWriter('./hist/vgg16/lr', sess.graph)

        # @TODO  restore parameters !!
        print("Time for initializing variables:",
              "{0:.2f}".format(time.time()-global_start), "sec")

        # Restore parameters from trained model
        ts = time.time()
        try:
            saver.restore(sess, saved_model_filepath)
        except Exception as err:
            print("Error in restoring model: ", err)

        print("Time for restoring params",
              "{0:.2f}".format(time.time()-ts), "sec")

        # Starting time for learning
        FatalError = False
        t1 = time.time()
        # Total number of steps/batches for writing event logs, e.g., total steps = training_epochs * num_batch
        log_writer_step = 0

        for epoch in range(training_epochs):
            print("\nStarting epoch", epoch)
            epoch_start = time.time()    # Start time for an epoch
            train_accuracy_list = []
            epoch_loss = []

            sess.run(dataset_init_op)
            num_batch = int(training_dataset_size//batch_size)

            for i in range(num_batch):
                start = time.time()    # Start time for a batch
                try:
                    if i % display_step:
                        sess.run(optimizer)  # training \

                    else:  # accuracy check
                        _, loss_print, train_accuracy = sess.run(
                            [optimizer, loss, accuracy])
                        print("epoch:{},batch:{}/{}".format(epoch, i, num_batch), "loss:{:.4f}".format(loss_print),
                              "acc:{0:.4f}".format(train_accuracy), "ex-time: {0:.2f}sec".format(time.time()-start))

                        writer_loss.add_summary(sess.run(merged_history, feed_dict={
                                                accuracy_history: loss_print}), log_writer_step)
                        writer_lr.add_summary(sess.run(merged_history, feed_dict={
                                              accuracy_history: sess.run(learning_rate)}), log_writer_step)
                        writer_train.add_summary(sess.run(merged_history, feed_dict={
                                                 accuracy_history: train_accuracy * 100}), log_writer_step)

                        train_accuracy_list.append(train_accuracy)
                        epoch_loss.append(loss_print)

                        log_writer_step = log_writer_step + 1

                except Exception as err:
                    print(
                        "Range-Error at epoch:{},batch{}/{}".format(epoch, i, num_batch))
                    print(err)
                    FatalError = True
                    break

            avg_train = np.mean(train_accuracy_list)
            avg_loss = np.mean(epoch_loss)

            # Save trained model after an epoch
            savedPath = saver.save(sess, saved_model_filepath)
            print("Model saved at: ", savedPath, ", after epoch", epoch)
            print("Epoch:", epoch, ", Average loss:", avg_loss,
                  ", Average training accuracy:", avg_train)
            if FatalError:
                break

        #label .failure
        savedPath = saver.save(sess, saved_model_filepath)
        print("Model saved at: ", savedPath, ", after epoch", epoch)
        print("\n")
        # Ending time for learning
        t2 = time.time()

        print("Learning time: " + "{0:.2f}".format(t2-t1) + " sec")

        writer_train.close()
        writer_loss.close()
        writer_lr.close()

    print("\nTraining finished!")
