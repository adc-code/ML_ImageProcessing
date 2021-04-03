#
# MNIST_NN_SoftMax.py 
#
# --> MNIST + Neural Network + SoftMax Function
#
# Script used to experiment a bit with the MNIST dataset with tensorflow


# for performance monitoring
import time


tic = time.perf_counter ()

import tensorflow as tf

toc = time.perf_counter ()
loadTime = toc - tic
print (f'Time to import tensorflow: {loadTime:0.4f} seconds')


from tensorflow import keras
import numpy as np


tic = time.perf_counter ()

data = keras.datasets.mnist
(x_train, Y_train), (x_test, Y_test) = data.load_data()

toc = time.perf_counter ()
loadTime = toc - tic
print (f'Time to load data: {loadTime:0.4f} seconds')

model = keras.Sequential()
model.add (keras.layers.Flatten())
model.add (keras.layers.Dense (units = 64, activation = 'relu'))
model.add (keras.layers.Dense (units = 16, activation = 'relu'))
model.add (keras.layers.Dense (units = 64, activation = 'relu'))
model.add (keras.layers.Dense (units = 10, activation = 'softmax'))

model.compile (optimizer = 'Nadam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

for i in range(1,11):
    model.fit (x_train, Y_train, epochs = i)
    losses, acc = model.evaluate (x_test, Y_test)
    print (i, 'Accuracy = ', acc) 

#img = cv2.imread('testpic.jpg', 0)    #'0' converts the img to gray
#img = cv2.resize(img, (28,28))
#prediction = model.predict([[img]])


