#
# FashionMNIST_Test.py 
#
# --> Fashion MNIST + Test...
#
# Script used to experiment a bit with the FashionMNIST dataset with tensorflow.
# 

SelectedModel = 2

ModelData = [ { 'ModelFile':    'FashionMNIST_Model_NN.h5', 
                'ModelDescStr': 'Simple Neural Network'         },
              { 'ModelFile':    'FashionMNIST_Model_CNN_1.h5',
                'ModelDescStr': 'Convolutional Neural Network - 3 Layers' },
              { 'ModelFile':    'FashionMNIST_Model_CNN_2.h5', 
                'ModelDescStr': 'Convolutional Neural Network - More Layers' },
              { 'ModelFile':    'FashionMNIST_Model_CNN_LeNet_1.h5', 
                'ModelDescStr': 'LeNet 5 - atanh - rbf' }, 
              { 'ModelFile':    'FashionMNIST_Model_CNN_LeNet_2.h5', 
                'ModelDescStr': 'LeNet 5 - relu - softmax' } ]

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'gamma': self.gamma
        })
        return config



# if needed convert the image to correct data format
def prepImage (img, imgSize):

    # first convert it to a np array from a PIL image
    img = np.array (img)

    # from 28 x 28 to 28 x 28 x 1
    img = tf.expand_dims (img, -1) 

    # next normalize it       
    img = tf.divide (img, 255)  

    # resize to the input of the pretrained neural net.
    img = tf.image.resize (img, [imgSize, imgSize])

    # reshape to add batch dimension
    img = tf.reshape(img, [1, imgSize, imgSize, 1])

    return img 


ItemNames = [ 'T-shirt-Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot' ]


# load the data...
(x_train_all, y_train_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data ()

if SelectedModel == 0:
    x_test = x_test / 255
elif SelectedModel == 1 or SelectedModel == 2:
    x_test = x_test / 255
    x_test = tf.expand_dims (x_test, -1)
    x_test = tf.image.resize (x_test, [28, 28])
elif SelectedModel == 3 or SelectedModel == 4:
    #x_test  = x_test / 128.0 - 1.0
    x_test = x_test / 255
    x_test = tf.expand_dims (x_test, -1)
    x_test = tf.image.resize (x_test, [28, 28])
    x_test = tf.pad (x_test, [ [0,0], [2,2], [2,2], [0,0] ])
    x_test = tf.cast (x_test, dtype=tf.float32)

# load the model as well...
if SelectedModel in [ 0, 1, 2 ]:
    model = keras.models.load_model (ModelData[SelectedModel]['ModelFile'])
else:
    model = keras.models.load_model (ModelData[SelectedModel]['ModelFile'], custom_objects={'RBFLayer': RBFLayer})
model.summary ()


# make the prediction
predProbs = model.predict (x_test)

# compute the confusion matrix
countResults = [ [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],   # 0
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],   # 1
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],   # 2
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],   # 3
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],   # 4
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],   # 5
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],   # 6
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],   # 7
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],   # 8
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] ]  # 9 

CorrectCount   = 0;
IncorrectCount = 0;

for i in range(len(x_test)):
    predValue = np.argmax (predProbs[i])
    countResults [y_test[i]][predValue] += 1

    if predValue == y_test[i]:
        CorrectCount += 1
    else:
        IncorrectCount += 1   

for i in range(10):
    print (ItemNames[i], countResults[i])

print ('Correct Count:   ', CorrectCount)
print ('Incorrect Count: ', IncorrectCount)
print ('Accuracy:        ', CorrectCount / (CorrectCount + IncorrectCount))


plt.figure ()
ax = sns.heatmap (countResults, annot=True, fmt='g', cmap="YlGnBu", xticklabels=ItemNames, yticklabels=ItemNames)
plt.title ('FashionMNIST - Classification Results - ' + ModelData[SelectedModel]['ModelDescStr'])
plt.show ()


