#
# FashionMNIST_Train_NN.py 
#
# --> Fashion MNIST + Training + (simple) Neural Network...
#
# Script used to experiment a bit with the FashionMNIST dataset with tensorflow.
# The purpose of this script is to train a model using the a simple neural
# network with the dataset.  It outputs a trained model used by the
# test script.
# 


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow import keras



# Model / data parameters
EPOCHS              = 20
VALIDATION_SET_SIZE = 5000
MODEL_OUTPUT_FILE   = 'FashionMNIST_Model_NN.h5'



#
# BuildModel: utility function to build the model...
#
def BuildModel ():

    # this model is heavily influenced from the 'hands on machine learning' book...
    model = keras.Sequential ()

    model.add (keras.layers.Flatten (input_shape=[28,28]))
    model.add (keras.layers.Dense (units = 300, activation = 'relu'))
    model.add (keras.layers.Dense (units = 100, activation = 'relu'))
    model.add (keras.layers.Dense (units = 10, activation = 'softmax'))
     
    model.compile (optimizer = keras.optimizers.SGD(lr=1e-3), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    return model



# load the data...
(x_train_all, y_train_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data ()


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

# train the model...
tic = time.perf_counter ()
trainHistory = model.fit (x_train, y_train, epochs = EPOCHS, validation_data = (x_valid, y_valid))
toc = time.perf_counter ()
trainTime = toc - tic
print (f'Time perform training...: {trainTime:0.4f} seconds\n\n')

# save the results
print (f'Writing model out to: {MODEL_OUTPUT_FILE}\n\n')
model.save (MODEL_OUTPUT_FILE)

# and evalute with the test set
results = model.evaluate (x_test, y_test)
print ('test loss, test acc: ', results)


# make a graph
sns.set_theme()
ax = sns.lineplot (data=trainHistory.history, markers=True, legend='auto')
ax.set_title ('Training Accuracy and Loss')
ax.set_xlabel ('Epoch')
plt.show()



