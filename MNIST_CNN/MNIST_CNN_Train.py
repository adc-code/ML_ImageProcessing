#
# MNIST_CNN_Train.py 
#
# --> MNIST + Convolutional Neural Network + For Training the Model...
#
# Script used to experiment a bit with the MNIST dataset with tensorflow.
# The purpose of this script is to train a model using the a few simple
# convolutional neural networks with the MNIST dataset.  It outputs a trained 
# model used by the test script (which is used to make predictions on test
# images).
# 


# Some key parameters used in the script...
MODEL_OUTPUT_FILE = 'MNIST_CNN_Model_1.h5'
MAX_EPOCHS        = 15
BATCH_SIZE        = 128

# Model / data parameters
NUM_CLASSES       = 10
INPUT_SHAPE       = (28, 28, 1)


# packages used...
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print ("x_train shape:", x_train.shape)
print (x_train.shape[0], "train samples")
print (x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical (y_train, NUM_CLASSES)
y_test  = keras.utils.to_categorical (y_test, NUM_CLASSES)


def BuildModel (modelType):

    model = keras.Sequential ()

    if modelType == 1:

        model.add (layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=INPUT_SHAPE))
        model.add (layers.MaxPooling2D((2, 2)))
        model.add (layers.Flatten())
        model.add (layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add (layers.Dense(10, activation='softmax'))

        opt = keras.optimizers.SGD (lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    elif modelType == 2:

        model.add (layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=INPUT_SHAPE))
        model.add (layers.BatchNormalization())
        model.add (layers.MaxPooling2D((2, 2)))
        model.add (layers.Flatten())
        model.add (layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add (layers.Dense(10, activation='softmax'))

        opt = keras.optimizers.SGD (lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    elif modelType == 3:
     
        model.add (keras.Input (shape=INPUT_SHAPE))
        model.add (layers.Conv2D (32, kernel_size=(3, 3), activation='relu'))
        model.add (layers.MaxPooling2D (pool_size=(2, 2)))
        model.add (layers.Conv2D (64, kernel_size=(3, 3), activation='relu'))
        model.add (layers.Conv2D (64, kernel_size=(3, 3), activation='relu'))
        model.add (layers.MaxPooling2D (pool_size=(2, 2)))
        model.add (layers.Flatten ())
        model.add (layers.Dropout (0.5))
        model.add (layers.Dense (NUM_CLASSES, activation='softmax'))
 
        model.compile (loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



model = BuildModel (3)

model.summary ()


tic = time.perf_counter ()

trainHistory = model.fit (x_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_split=0.1)

toc = time.perf_counter ()
trainTime = toc - tic
print (f'Time perform training...: {trainTime:0.4f} seconds\n\n')


score = model.evaluate (x_test, y_test, verbose=0)
print (f'Test loss: {score[0]}')
print (f'Test accuracy: {score[1]}')

print (f'Writing model out to: {MODEL_OUTPUT_FILE}\n\n')
model.save (MODEL_OUTPUT_FILE)


plt.figure ()
plt.plot (trainHistory.history ['loss'], label='Training Loss')
plt.plot (trainHistory.history ['val_loss'], label='Validation Loss')
plt.plot (trainHistory.history ['accuracy'], label='Training Accuracy')
plt.plot (trainHistory.history ['val_accuracy'], label='Validation Accuracy')
plt.title ('Accuracy & Loss over Training')
plt.legend (bbox_to_anchor=(1.04,1), loc='upper left')
plt.show ()


