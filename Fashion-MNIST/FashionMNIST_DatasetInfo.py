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

             
import tensorflow as tf
from tensorflow import keras


data = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = data.load_data()


count_TrainSet = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
count_TestSet  = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]


for y in y_train:
    count_TrainSet [y] += 1

for y in y_test:
    count_TestSet [y] += 1

print (f'{"Item":12} {"Train-Set":9} {"Test-Set":8}')
for i in range (len(ItemNames)):
    if ItemNames[i]['Index'] != -1:
        print (f"{ItemNames[i]['Name']:12} {count_TrainSet[ItemNames[i]['Index']]:9} {count_TestSet[ItemNames[i]['Index']]:8}")


