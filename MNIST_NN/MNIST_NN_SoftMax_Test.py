#
# MNIST_NN_SoftMax_Test.py 
#
# --> MNIST + Neural Network + SoftMax Function + Test (predict)
#
# Script used to experiment a bit with the MNIST dataset with a simple neural 
# network with tensorflow.  This program tries to predict the input number from
# a image.
#


# Load key packages...
import sys
import numpy as np
from PIL import Image
import MNIST_OutputNumbers


# Check command line parameters...
if len (sys.argv) != 2:
    print ('Usage Error!')
    print ()
    print (f'{sys.argv[0]}  <FileName>')
    sys.exit ()


print ('Loading tensorflow...')
import tensorflow as tf
from tensorflow import keras


# File to be predicted...
ImgFileName = sys.argv [1]


#
# prepImage... prepare the image so it has the same dimensions as those used by
#              the neural network, otherwise TF will complain.  Empty dimensions
#              are added to include the batchsize and number of channels, even 
#              if they are not used.
# 
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


# Load an already trained model...
model = keras.models.load_model ('MNIST_NN_SoftMax_Model_35.h5')


# Uncomment for debugging... display the model's architecture
# model.summary ()


# load the image and prep it before it is used...
img = Image.open (ImgFileName).convert ('L')
img = prepImage (img, 28)


# Make the prediction...
preds = model.predict (img)


# Output the prediction percentages...
print ()
print (' -=- Prediction Percent Values -=- ')
for i, p in enumerate (*preds):
    print (f'{i} --> {100*p:.4} %') 


# Output a large and fancy representation of the 'best' guess based 
# on the prediction values
print ()
print (' -=- Best Guess -=- ')
label = np.argmax (preds)
print (MNIST_OutputNumbers.MakeFancyNumber (label))



