#
# MNIST_NN_SoftMax_Train.py 
#
# --> MNIST + Neural Network + SoftMax Function + For Training the Model...
#
# Script used to experiment a bit with the MNIST dataset with tensorflow.
# The purpose of this script is to train a model using the a simple neural
# network with the MNIST dataset.  It outputs a trained model used by the
# test script.
# 


VALIDATION_SET_SIZE = 5000
MODEL_OUTPUT_FILE   = 'MNIST_NN_SoftMax_Model.h5'


#
# BuildModel: utility function to build the model...
#
def BuildModel ():
    model = keras.Sequential ()
    model.add (keras.layers.Flatten (input_shape=[28,28]))
    model.add (keras.layers.Dense (units = 200, activation = 'relu'))
    model.add (keras.layers.Dense (units = 50, activation = 'relu'))
    model.add (keras.layers.Dense (units = 10, activation = 'softmax'))

    model.compile (optimizer = 'Nadam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# load tensorflow... (note can take a moment)
print ('Importing tensorflow2...')
import tensorflow as tf
from tensorflow import keras

# load the rest
print ('Importing remaining packages...')
import numpy as np
import matplotlib.pyplot as plt
import time


# load the data...
(x_train_all, y_train_all), (x_test, y_test) = keras.datasets.mnist.load_data ()


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

# Train the model...
tic = time.perf_counter ()
trainHistory = model.fit (x_train, y_train, epochs = 30, validation_data = (x_valid, y_valid))
toc = time.perf_counter ()
trainTime = toc - tic
print (f'Time perform training...: {trainTime:0.4f} seconds\n\n')


# Look at the test data...
print (f'Evaluate the test data...')
results = model.evaluate (x_test, y_test)
print ('test loss, test acc: ', results)
print ()

print (f'Writing model out to: {MODEL_OUTPUT_FILE}\n\n')
model.save (MODEL_OUTPUT_FILE)


plt.figure ()
plt.plot (trainHistory.history ['loss'])
plt.plot (trainHistory.history ['val_loss'])
plt.plot (trainHistory.history ['accuracy'])
plt.plot (trainHistory.history ['val_accuracy'])
plt.show ()



