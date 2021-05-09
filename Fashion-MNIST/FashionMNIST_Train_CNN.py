#
# FashionMNIST_Train_CNN.py 
#
# --> Fashion MNIST + Training + Convolutional Neural Network...
#
# Script used to experiment a bit with the Fashion-MNIST dataset with tensorflow.
# The purpose of this script is to train a model using the a (rather simple) 
# convolutional neural network with the MNIST dataset.  It outputs a trained model 
# used by the test script.
# 


import tensorflow as tf
from tensorflow import keras

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time


EPOCHS              = 20
IMG_WIDTH           = 28
IMG_HEIGHT          = 28
IMG_DEPTH           = 1   # (one channel since its grey)
VALIDATION_SET_SIZE = 5000
MODEL_OUTPUT_FILE   = 'FashionMNIST_Model_CNN_.h5'


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

    model.add (keras.Input (shape=inputShape))
    model.add (keras.layers.Conv2D (32, kernel_size=(3, 3), activation='relu'))
    model.add (keras.layers.MaxPooling2D (pool_size=(2, 2)))
    model.add (keras.layers.Conv2D (64, kernel_size=(3, 3), activation='relu'))
    model.add (keras.layers.Conv2D (64, kernel_size=(3, 3), activation='relu'))
    model.add (keras.layers.MaxPooling2D (pool_size=(2, 2)))
    model.add (keras.layers.Flatten ())
    model.add (keras.layers.Dropout (0.5))
    model.add (keras.layers.Dense (10, activation='softmax'))
    model.compile (optimizer = 'Nadam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# load the data...
(x_train_all, y_train_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data ()

if keras.backend.image_data_format() == "channels_first":
    x_train_all = x_train_all.reshape ((x_train_all.shape[0], 1, 28, 28))
    x_test = x_test.reshape ((x_test.shape[0], 1, 28, 28))
 
# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
    x_train_all = x_train_all.reshape ((x_train_all.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# for debugging...
# print (f'keras.backend: {keras.backend.image_data_format()}')


# Split the loaded training data into train and validation sets.  Also normalize the data at the 
# same time
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




