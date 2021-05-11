#
# FashionMNIST_Train_CNN_LeNet.py 
#
# --> Fashion MNIST + Training + Convolutional Neural Network + LeNet Style...
#
# Script used to experiment a bit with the Fashion-MNIST dataset with tensorflow.
# The purpose of this script is to train a model using the LeNet convolutional 
# neural network with the Fashion MNIST dataset.  It outputs a trained model 
# used by the test script.  Note that this version uses relu's instead of atanh's
# for the activation functions and softmax instead of RBF.
# 


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time


EPOCHS              = 20
IMG_WIDTH           = 28
IMG_HEIGHT          = 28
IMG_DEPTH           = 1   # (one channel since its grey)
VALIDATION_SET_SIZE = 5000
MODEL_OUTPUT_FILE   = 'FashionMNIST_Model_CNN_LeNet_2.h5'



#
# BuildModel: utility function to build the model...
#
def BuildModel ():

    model = keras.Sequential ()

    # this code to deal with channels first/last was taken from https://www.pyimagesearch.com/
    inputShape = (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)
    chanDim = -1

    if keras.backend.image_data_format() == 'channels_first':
        inputShape = (IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH)
        chanDim = 1

    model.add (keras.layers.Conv2D (6, (5,5), padding='same', activation='relu', input_shape=(32, 32, 1)))
    model.add (keras.layers.AveragePooling2D ((2, 2), strides=2))
    model.add (keras.layers.Conv2D (16, (5,5), padding='same', activation='relu'))
    model.add (keras.layers.AveragePooling2D ((2, 2), strides=2))
    model.add (keras.layers.Conv2D (120, (5,5), activation='relu'))
    model.add (keras.layers.Flatten ())
    model.add (keras.layers.Dense(84, activation=tf.nn.tanh))
    model.add (keras.layers.Dense(10,  activation=tf.nn.softmax))

    model.compile (optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# load the data...
(x_train_all, y_train_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data ()


# for debugging...
# print (f'keras.backend: {keras.backend.image_data_format()}')

if keras.backend.image_data_format() == "channels_first":
    x_train_all = x_train_all.reshape ((x_train_all.shape[0], 1, 28, 28))
    x_test = x_test.reshape ((x_test.shape[0], 1, 28, 28))
else:
    # otherwise, we are using "channels last" ordering, so the design
    # tensor shape should be: num_samples x rows x columns x depth
    x_train_all = x_train_all.reshape ((x_train_all.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))



# Split the loaded training data into train and validation sets.  Also normalize the data at the 
# same time

x_train_all = tf.pad (x_train_all, [ [0,0], [2,2], [2,2], [0,0] ])
x_train_all = tf.cast (x_train_all, dtype=tf.float32)

x_test = tf.pad (x_test, [ [0,0], [2,2], [2,2], [0,0] ])
x_test = tf.cast (x_test, dtype=tf.float32)


x_valid = x_train_all [:VALIDATION_SET_SIZE] / 255.0
x_train = x_train_all [VALIDATION_SET_SIZE:] / 255.0
x_test  = x_test / 255.0
y_valid = y_train_all [:VALIDATION_SET_SIZE] 
y_train = y_train_all [VALIDATION_SET_SIZE:] 


print (f'Training data size: {x_train.shape} ')
print (f'Validation data size: {x_valid.shape} ')
print (f'Testing data size: {x_test.shape}\n')


# Create a basic model instance
model = BuildModel ()

# Display the model's architecture
model.summary ()

tic = time.perf_counter ()
trainHistory = model.fit (x_train, y_train, epochs = EPOCHS, validation_data = (x_valid, y_valid))
toc = time.perf_counter ()
trainTime = toc - tic
print (f'Time perform training...: {trainTime:0.4f} seconds\n\n')

print (f'Writing model out to: {MODEL_OUTPUT_FILE}\n\n')
model.save (MODEL_OUTPUT_FILE)

results = model.evaluate (x_test, y_test)
print ('test loss, test acc: ', results)

# make a graph
sns.set_theme()
ax = sns.lineplot (data=trainHistory.history, markers=True, legend='auto')
ax.set_title ('Training Accuracy and Loss')
ax.set_xlabel ('Epoch')
plt.show ()


