#
# GenerateMNISTSampleSetImage.py
#
# Script used to generate a sample set image from the MNIST dataset.  Note that the 
# selections are done purely at random.  The purpose of this script is just to view some
# elements from the dataset.
#
# Usage: > python GenerateMNISTSampleSetImage.py  <Horiztonal Count>  <Vertical Count>  <OutputFile>
#


import sys
import time
import random
from sklearn.datasets import fetch_openml
from PIL import Image


if len (sys.argv) != 4:
        print ('Error: incorrect number of parameters!')
        print ()
        print ('Usage:', sys.argv[0],'  <Horiztonal Count>  <Vertical Count>  <OutputFile>')
        sys.exit ()


HorizCount = int (sys.argv [1])
VertCount  = int (sys.argv [2])

ImgWidth  = 28
ImgHeight = 28

# Create the image
combinedImg = Image.new ('L', (HorizCount * ImgWidth, VertCount * ImgHeight), 255)

# Uncomment for testing...
tic = time.perf_counter()

mnist = fetch_openml ('mnist_784', version=1)

toc = time.perf_counter()
print (f'Time to preform Fetch: {toc - tic:0.4f} seconds')

X, y = mnist['data'], mnist['target']

# select elements and copy them into the target image
for i in range (VertCount):
    for j in range (HorizCount):
        digit = X[random.randint (0, X.shape[0])]
        digitImg = Image.fromarray (digit.reshape (28, 28))

        combinedImg.paste (digitImg, (ImgWidth*j, ImgHeight*i))

# finally, write out the image
combinedImg.save (sys.argv[3])


