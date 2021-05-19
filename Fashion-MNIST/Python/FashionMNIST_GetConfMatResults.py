#
# FashionMNIST_GetConfMatResults.py 
#
# Script used to get the confusion matrix results that can be further graphed with
# another script.
# 

SelectedModel = 2

MAX_IDLIST_SIZE = 5

OutputFileName          = 'FashionMNIST-ConfMatData.json'
OutputImageDir          = 'FashionMNIST_OutputImages'
OutputImageFileBasename = 'FashionMNIST_Item_'
OutputImageFileExt      = '.png'

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
from PIL import Image
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


ItemNames = [ 'T-shirt-Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot' ]


# load the data...
(x_train_all, y_train_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data ()

if SelectedModel == 0:
    x_test_scaled = x_test / 255
elif SelectedModel == 1 or SelectedModel == 2:
    x_test_scaled = x_test / 255
    x_test_scaled = tf.expand_dims (x_test_scaled, -1)
    x_test_scaled = tf.image.resize (x_test_scaled, [28, 28])
elif SelectedModel == 3 or SelectedModel == 4:
    x_test_scaled = x_test / 255
    x_test_scaled = tf.expand_dims (x_test_scaled, -1)
    x_test_scaled = tf.image.resize (x_test_scaled, [28, 28])
    x_test_scaled = tf.pad (x_test_scaled, [ [0,0], [2,2], [2,2], [0,0] ])
    x_test_scaled = tf.cast (x_test_scaled, dtype=tf.float32)

# load the model as well...
if SelectedModel in [ 0, 1, 2 ]:
    model = keras.models.load_model (ModelData[SelectedModel]['ModelFile'])
else:
    model = keras.models.load_model (ModelData[SelectedModel]['ModelFile'], custom_objects={'RBFLayer': RBFLayer})
model.summary ()


# make the prediction
predProbs = model.predict (x_test_scaled)


countDetails = []
for i in range(10):
    countDetailsRow = []
    for j in range(10):
        countDetailsItem = {
                         'i':          i,
                         'j':          j,
                         'Count':      0,
                         'Actual':     ItemNames[i],
                         'Predicted':  ItemNames[j],
                         'IDs':        [],
                         'PredValues': []
                     }
        countDetailsRow.append (countDetailsItem)
    countDetails.append (countDetailsRow)



CorrectCount   = 0;
IncorrectCount = 0;

#
# gather all the details for each combination
#
for i in range(len(x_test)):
    predValue = np.argmax (predProbs[i])

    countDetails[y_test[i]][predValue]['IDs'].append (i)
    countDetails[y_test[i]][predValue]['Count'] += 1
    countDetails[y_test[i]][predValue]['PredValues'].append (predProbs[i])

    if predValue == y_test[i]:
        CorrectCount += 1
    else:
        IncorrectCount += 1   

#
# remove any ids and predictions values more than MAX_IDLIST_SIZE
#
for i in range(len(countDetails)):
    for j in range(len(countDetails[i])):
        if len (countDetails[i][j]['IDs']) > MAX_IDLIST_SIZE:
            countDetails[i][j]['IDs']        = countDetails[i][j]['IDs'][:MAX_IDLIST_SIZE]
            countDetails[i][j]['PredValues'] = countDetails[i][j]['PredValues'][:MAX_IDLIST_SIZE]


#
# output the json file...
#
outputFile = open (OutputFileName, 'w')

outputFile.write ('[\n')

for i in range(len(countDetails)):
    for j in range(len(countDetails[i])):

        outputFile.write ('{\n')
        
        outputFile.write ('    "i": ' + str(countDetails[i][j]['i']) + ',\n')
        outputFile.write ('    "j": ' + str(countDetails[i][j]['j']) + ',\n')

        outputFile.write ('    "Count": ' + str(countDetails[i][j]['Count']) + ',\n')

        outputFile.write ('    "Actual":    "' + countDetails[i][j]['Actual']    + '",\n')
        outputFile.write ('    "Predicted": "' + countDetails[i][j]['Predicted'] + '",\n')

        # since IDs is a list of items, got to print each value separately
        outputFile.write ('    "IDs": [ ')
        for k in range (len(countDetails[i][j]['IDs'])):
            outputFile.write (str(countDetails[i][j]['IDs'][k]))
            if k != len(countDetails[i][j]['IDs'])-1:
                outputFile.write (', ')
        outputFile.write (' ], \n')

        # the prediction values is a list of lists... this code is ugly but is fine for now...
        outputFile.write ('    "PredictionValues": [\n')
        for k in range (len(countDetails[i][j]['PredValues'])):

            outputFile.write ('        [ ')
            for l in range (len(countDetails[i][j]['PredValues'][k])):
                value = f" {(100 * countDetails[i][j]['PredValues'][k][l]):.3f}"
                outputFile.write (value)
                if l != len(countDetails[i][j]['PredValues'][k])-1:
                    outputFile.write (',')

            outputFile.write (' ]')

            if k != len(countDetails[i][j]['PredValues'])-1:
                outputFile.write (',')
            outputFile.write ('\n')

        outputFile.write ('                         ]\n')

        if i == len(countDetails)-1 and j == len(countDetails[i])-1:
            outputFile.write ('}\n')
        else:
            outputFile.write ('},\n')
       
 
outputFile.write (']\n')
outputFile.close ()

print ('Correct Count:   ', CorrectCount)
print ('Incorrect Count: ', IncorrectCount)
print ('Accuracy:        ', CorrectCount / (CorrectCount + IncorrectCount))

WriteImages = False

if WriteImages:
    for i in range(len(countDetails)):
        for j in range(len(countDetails[i])):

            for k in range(len(countDetails[i][j]['IDs'])):
                FashionItem  = x_test[ countDetails[i][j]['IDs'][k] ]
                FashionImage = Image.fromarray ( FashionItem.reshape (28,28) )
                FashionImage = FashionImage.resize ( (42,42) )
                FashionImage.save (OutputImageDir + '/' + OutputImageFileBasename + str(countDetails[i][j]['IDs'][k]) + OutputImageFileExt)


for i in range(len(countDetails)):
    outputString = countDetails[i][j]['Actual'] + '  '
    for j in range(len(countDetails[i])):
        outputString += str(countDetails[i][j]['Count']) + ' '
    print (outputString)


