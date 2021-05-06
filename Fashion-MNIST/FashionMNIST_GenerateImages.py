#
# FashionMNIST_GenerateImages.py 
#
# Script used to dump out some images from the Fashion MNIST dataset
#


ItemNames = [ { 'Index': -1, 'Name': 'Any-All'     },
              { 'Index':  0, 'Name': 'T-shirt-Top' },
              { 'Index':  1, 'Name': 'Trouser'     },
              { 'Index':  2, 'Name': 'Pullover'    },
              { 'Index':  3, 'Name': 'Dress'       },
              { 'Index':  4, 'Name': 'Coat'        },
              { 'Index':  5, 'Name': 'Sandal'      },
              { 'Index':  6, 'Name': 'Shirt'       },
              { 'Index':  7, 'Name': 'Sneaker'     },
              { 'Index':  8, 'Name': 'Bag'         },
              { 'Index':  9, 'Name': 'AnkleBoot'   } ]

             
import sys
import numpy as np
import random
from PIL import Image

# for performance monitoring
import time

# import tensorflow stuff... can take a moment on a slower computer
tic = time.perf_counter ()
import tensorflow as tf
from tensorflow import keras
toc = time.perf_counter ()
loadTime = toc - tic
print (f'Time to import tensorflow: {loadTime:0.4f} seconds')


# report message if not enough command line parameters were specified...
if len (sys.argv) != 5:

    print ('Error: incorrect number of parameters!')
    print ()
    print ('Usage:', sys.argv[0],'  <Horiztonal Count>  <Vertical Count>  <Fashion Item>  <BaseOutputName> ')
    print ()
    print ('    Where <Fashion Item> can have the following values to select specific fashion items:')
    for item in ItemNames:
        print (f"         {item['Index']:2} -> {item['Name']}")
    print ()

    sys.exit ()


# Draw grid used to draw a grid around the numbers
DrawGrid  = False #True

ImgWidth  = 28
ImgHeight = 28

# should really check that these are valid... 
HorizCount     = int (sys.argv [1])
VertCount      = int (sys.argv [2])
SelectedItem   = int (sys.argv [3])
BaseOutputName = sys.argv [4]

# load the dataset...
tic = time.perf_counter ()
data = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
toc = time.perf_counter ()
loadTime = toc - tic
print (f'Time to load data: {loadTime:0.4f} seconds')


# make a image to store everything...
if DrawGrid == True:
    combinedImg = Image.new ('L', (HorizCount * (ImgWidth + 2), VertCount * (ImgHeight + 2)), 255)
else:
    combinedImg = Image.new ('L', (HorizCount * ImgWidth, VertCount * ImgHeight), 255)


alreadyPickedIndices = []
for i in range (VertCount):
    for j in range (HorizCount):
  
        # select a random number and make sure we don't repeat 
        while (True):
            randomIndex = random.randint (0, x_train.shape[0])

            # first check if selected item is of the desired type
            if y_train [ randomIndex ] != SelectedItem and SelectedItem != -1:
                continue

            # and if it was not selected, add it to the list
            if randomIndex not in alreadyPickedIndices:
                alreadyPickedIndices.append (randomIndex)
                break

        fashionItem = x_train [ alreadyPickedIndices[-1] ]
        fashionImg = Image.fromarray (fashionItem.reshape (ImgWidth, ImgHeight))

        # add the image into the set
        if DrawGrid == True:
            combinedImg.paste (fashionImg, ( (ImgWidth+2)*j + 1, (ImgHeight+2)*i + 1))
        else:
            combinedImg.paste (fashionImg, (ImgWidth*j, ImgHeight*i))

# finally save the image
combinedImg.save (BaseOutputName + '_' + ItemNames[SelectedItem+1]['Name'] + '.jpg')



