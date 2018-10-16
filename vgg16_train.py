

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


train_label_file = 'labels.txt'
test_label_file = 'ILSVRC2012_test_ground_truth.txt'
validation_label_file = 'ILSVRC2012_validation_ground_truth.txt'


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


#training_dataset_size = len(os.listdir(training_dataset_folder))
testing_dataset_size = len(os.listdir(testing_dataset_folder))
validation_dataset_size = len(os.listdir(validation_dataset_folder))

#print("Number of training image folders: ", training_dataset_size)


# Initialize parameters
training_epochs = 50
batch_size = 64
input_width = 224
input_depth = 3
output_classes = len(training_GT)

learning_rate = 0.001
VGG_MEAN = [123.68, 116.779, 103.939]
means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])

#print("Testing images: ", testing_dataset_size)
#print("Validation images: ", validation_dataset_size)

training_image_folders = os.listdir(training_dataset_folder)
for each in training_image_folders:
    if ".tar" in each:
        training_image_folders.remove(each)
print("Training folders : ", len(training_image_folders))


all_training_images = []
all_training_image_labels = []


def getListOfImages():
    global all_training_images
    global all_training_image_labels
    
    for f in range(len(training_image_folders)):
        images = os.listdir(training_dataset_folder + training_image_folders[f])
        for im in range(len(images)):
            image_path = training_dataset_folder + training_image_folders[f] + "/" + images[im]
            all_training_images.append(image_path)
            all_training_image_labels.append(f + 1)
            
    return all_training_images, all_training_image_labels
            
            
def shuffleImagesPath(imagesPathArray, imagesLabelsArray):
    print("Number of total training images: ", len(imagesPathArray))
    for i in range(0, len(imagesPathArray)):
        randomIndex1 = randint(0, len(imagesPathArray)-1)
        randomIndex2 = randint(0, len(imagesPathArray)-1)
        imagesPathArray[randomIndex1], imagesPathArray[randomIndex2] = imagesPathArray[randomIndex2], imagesPathArray[randomIndex1]
        imagesLabelsArray[randomIndex1], imagesLabelsArray[randomIndex2] = imagesLabelsArray[randomIndex2], imagesLabelsArray[randomIndex1]
    
    return imagesPathArray, imagesLabelsArray


def getNumber(label):
    return np.eye(output_classes, dtype=np.float32)[int(label)]
    

def decode_image(pathToImage):
    #image = imread(str(pathToImage), mode='RGB')
    #imarray = imresize(image, (input_width, input_width))
    #print("\nStarting reading image - ", pathToImage)
    imageContents = tf.read_file(str(pathToImage))
    #print("Starting decoding image")
    image = tf.image.decode_jpeg(imageContents, channels=3)
    #print("Starting resizing image")
    resized_image = tf.image.resize_images(image, [input_width, input_width])
    imarray = resized_image.eval()
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
    
              
# Return the batch of training data for next run
def get_next_batch_of_training_images(imagesPathArray, image_labels):
     
    dataset = np.ndarray(shape=(0, input_width, input_width, input_depth), dtype=np.float32)
    labels = np.ndarray(shape=(0, output_classes), dtype=np.float32)
    
    #dataset = tf.placeholder(tf.float32, shape=[None, input_width, input_width, input_depth])
    #labels = tf.placeholder(tf.float32, shape=[None, output_classes])

    for i in range(len(imagesPathArray)):

        try:
         
            imarray = decode_image(imagesPathArray[i])
            imlabel = getNumber(image_labels[i])
            #image = _parse_function(imagesPathArray[i])
            #imarray = training_preprocess(image)
            #print("Starting making array of image")
            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array([imlabel], dtype=np.float32)
            #appendingImageArray = tf.convert_to_tensor([imarray], dtype=tf.float32)
            #appendingNumberLabel = tf.convert_to_tensor([imlabel], dtype=tf.float32)
            
            #print(appendingImageArray.shape, appendingNumberLabel.shape)
            
            dataset = np.append(dataset, appendingImageArray, axis=0)
            labels = np.append(labels, appendingNumberLabel, axis=0)
            #dataset = tf.concat([dataset, appendingImageArray], axis=0)
            #labels = tf.concat([labels, appendingNumberLabel], axis=0)
            
        except Exception as err:
            print("Unexpected image - ", imagesPathArray[i], ", skipping...", err)
            
    return dataset, labels
            
            
            
def get_testing_images(imagesPathArray, labels_array, type="testing"):

    dataset = np.ndarray(shape=(0, input_width, input_width, input_depth), dtype=np.float32)
    labels = np.ndarray(shape=(0, output_classes), dtype=np.float32)
    
    for i in range(len(imagesPathArray)):

        try:
            pathToImage = testing_dataset_folder + imagesPathArray[i]
            if type == "validation":
                pathToImage = validation_dataset_folder + imagesPathArray[i]
                
            #imarray = decode_image(pathToImage)
            image = _parse_function(pathToImage)
            imarray = val_preprocess(image)
            
            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array([getNumber(labels_array[i])], dtype=np.float32)
            
            dataset = np.append(dataset, appendingImageArray, axis=0)
            labels = np.append(labels, appendingNumberLabel, axis=0)
        
        except Exception as err:
            print("Unexpected image - ", imagesPathArray[i], " skipping...", err)
            
    return dataset, labels

    
            
# Initialize input and output
x = tf.placeholder(tf.float32, shape=[None, input_width, input_width, input_depth])
y = tf.placeholder(tf.float32, shape=[None, output_classes])


# model
vgg = vgg16.vgg16(x)
logits = vgg.logits


# Loss function
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
        
        # test and validation images
        test_image_names = os.listdir(testing_dataset_folder)
        val_image_names = os.listdir(validation_dataset_folder)
        
        
        for epoch in range(1, training_epochs + 1):
        
            train_accuracy_list = []
            test_accuracy_list = []
            loss_list = []
            
            print("\n")
            print("Starting epoch: ", epoch)
            
            
            all_training_images, all_training_image_labels = getListOfImages()
            all_training_images, all_training_image_labels = shuffleImagesPath(all_training_images, all_training_image_labels)
            print("Shuffling images for epoch", epoch)
          
                
            start_index = 0
            num_images = len(all_training_images)
            num_steps = round(num_images/batch_size)
            
        
            for j in range(num_steps):
        
                try:
                    print("\nStarting Epoch:", epoch, ", Step:", j + 1)
                             
                    try:
                        if j > 0:
                            saver.restore(sess, saved_model_filepath)
                    except Exception as e:
                        print(e)
                        
                    last_index = start_index + (batch_size - 1)
                
                    step_images = all_training_images[start_index:last_index]
                    step_labels = all_training_image_labels[start_index:last_index]
                    
                    #print("Getting images for Epoch:", epoch, ", Step:", j + 1)
                    batch_train_images, batch_train_labels = get_next_batch_of_training_images(step_images, step_labels)
                    training_data = {x: batch_train_images, y: batch_train_labels}
                    
                    #print("Starting optimizer for Epoch:", epoch, ", Step:", j + 1)
                    sess.run(optimizer, feed_dict=training_data)
                    
                    # logging
                    
                    #print("Getting accuracy and loss for Epoch:", epoch, ", Step:", j + 1)
                    train_accuracy = accuracy.eval(feed_dict=training_data)
                    loss_print = cost.eval(feed_dict=training_data)
                    
                    train_accuracy_list.append(train_accuracy)
                    loss_list.append(loss_print)
                    print("Epoch:", epoch, ", Step:", j + 1, ", Loss:", loss_print, ", Training Accuracy:", train_accuracy)
                    
                    # saving model
                    savedPath = saver.save(sess, saved_model_filepath)
                    #print("Model saved at: " , savedPath)
                    
                except Exception as err:
                    print("Error in training! Epoch:", epoch, ", Step:", j + 1, err)            

                start_index = start_index + batch_size
                
                    
            training_accuracy = np.mean(train_accuracy_list)
            loss_here = np.mean(loss_list)
            
            writer_train.add_summary(sess.run(merged_history, feed_dict={accuracy_history: training_accuracy}), epoch)
            writer_loss.add_summary(sess.run(merged_history, feed_dict={accuracy_history: loss_here}), epoch)
        
            
            
            start_index = 0
            num_images = len(test_image_names)
            num_steps = round(num_images/batch_size)
            
        
            for k in range(num_steps):            
                try:
                    last_index = start_index + (batch_size - 1)
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
            writer_test.add_summary(sess.run(merged_history, feed_dict={accuracy_history: testing_accuracy}), epoch)
            
            print("Training accuracy: ", training_accuracy * 100, ", Loss: ", loss_here, "%, Testing accuracy: ", testing_accuracy * 100, "%, ", epoch, "th epoch.")
            
                
        print("\n")
        # Ending time
        t2 = time.time()
                
        # Save trained model
        savedPath = saver.save(sess, saved_model_filepath)
        print("Final model saved at: " , savedPath)
        
            
        print("Learning time: " + str(t2-t1) + " seconds")
        
        
        val_acc_list = []
        start_index = 0
        num_images = len(val_image_names)
        num_steps = round(num_images/batch_size)
            
            
        for m in range(num_steps):            
            try:
                last_index = start_index + (batch_size - 1)
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
        print("Validation accuracy: ", validation_accuracy * 100)
            

    



