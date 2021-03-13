#
# MNIST_FullClassifier_DecisionFuncResults.py
#
# Small utility program used to output some key elements from the 'cases of interest',
# that is the best and worst predicted cases.  Note that the SGD classifier is used 
# since it is much easier to optain bad results from it (this code is mainly for learning
# not production, so one of the simpliest and perhaps worst classifiers was used).  The
# output includes jpg images for the good/bad images along with a csv file with all
# the decision function values.
#

import numpy as np
import time
import sys

from PIL import Image

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier


#
# The cases of interest...
# 3 - 2,  3 - 5,  4 - 9, 5 - 3, 5 - 8, 8 - 3,  8 - 5, 9 - 7 
#
casesOfInterest = [
    { 'expectedValue': 3, 'predictedValue': 2, 'goodIndices': [], 'badIndices': [] },
    { 'expectedValue': 3, 'predictedValue': 5, 'goodIndices': [], 'badIndices': [] },
    { 'expectedValue': 4, 'predictedValue': 9, 'goodIndices': [], 'badIndices': [] },
    { 'expectedValue': 5, 'predictedValue': 3, 'goodIndices': [], 'badIndices': [] },
    { 'expectedValue': 5, 'predictedValue': 8, 'goodIndices': [], 'badIndices': [] },
    { 'expectedValue': 8, 'predictedValue': 3, 'goodIndices': [], 'badIndices': [] },
    { 'expectedValue': 8, 'predictedValue': 5, 'goodIndices': [], 'badIndices': [] },
    { 'expectedValue': 8, 'predictedValue': 9, 'goodIndices': [], 'badIndices': [] },
    { 'expectedValue': 9, 'predictedValue': 7, 'goodIndices': [], 'badIndices': [] }    
]


#
# fetch the data...
#
tic = time.perf_counter ()
mnist = fetch_openml ('mnist_784', version=1)
toc = time.perf_counter ()
loadTime = toc - tic
print (f'Time to preform Fetch: {loadTime:0.4f} seconds')


#
# split the data into test and train sets...
#
TRAIN_TEST_SPLIT = 10000
X = mnist ['data']
y = mnist ['target']
y = y.astype (np.uint8)

X_train, X_test = X[:(X.shape[0] - TRAIN_TEST_SPLIT)], X[(X.shape[0] - TRAIN_TEST_SPLIT):]
y_train, y_test = y[:(X.shape[0] - TRAIN_TEST_SPLIT)], y[(X.shape[0] - TRAIN_TEST_SPLIT):]


#
# fit the data using stochastic gradient descent (SGD)...
# 
clf = SGDClassifier (random_state=42)

tic = time.perf_counter()
clf.fit (X_train, y_train)
toc = time.perf_counter()
trainTime = toc - tic
print (f'Time to preform fit data: {trainTime:0.4f} seconds')


#
# go through everything and determine the indices of all numbers in the train set... perhaps this is too much
#
indices = [ [], [], [], [], [], [], [], [], [], [] ]
for i in range(len(y_train)):
    if y_train[i] == 0:
        indices[0].append (i)
    elif y_train[i] == 1:
        indices[1].append (i)
    elif y_train[i] == 2:
        indices[2].append (i)
    elif y_train[i] == 3:
        indices[3].append (i)
    elif y_train[i] == 4:
        indices[4].append (i)
    elif y_train[i] == 5:
        indices[5].append (i)
    elif y_train[i] == 6:
        indices[6].append (i)
    elif y_train[i] == 7:
        indices[7].append (i)
    elif y_train[i] == 8:
        indices[8].append (i)
    elif y_train[i] == 9:
        indices[9].append (i)


#
# Go through all the cases of interest and find the best and worst predictions...
#
for i, case in enumerate (casesOfInterest):

    print (f'Evaluating Case: {i}')
    
    indicesForExpectedValue = []
    indicesForPredictedValue = []

    for index in indices [ case['expectedValue'] ]:
        predTest = clf.predict (X_train[index].reshape(1, -1))
    
        if predTest == case ['predictedValue']:
            decValues = clf.decision_function (X_train [index].reshape(1, -1))
            indicesForPredictedValue.append ( [ index, decValues.tolist()[0][case ['predictedValue']] ] )
            
        elif predTest == case ['expectedValue']:
            decValues = clf.decision_function (X_train [index].reshape(1, -1))
            indicesForExpectedValue.append ( [ index, decValues.tolist()[0][case ['expectedValue']] ] )

    indicesForPredictedValue.sort (key=lambda x: x[1], reverse=True)
    indicesForExpectedValue.sort (key=lambda x: x[1], reverse=True)

    for i in range (10):
        case ['goodIndices'].append (indicesForExpectedValue[i][0])
        case ['badIndices'].append (indicesForPredictedValue[i][0])


#
# output the decision function results to a file and images for the best and worst cases... 
#
outputFileName = 'FullClassifier_DecisionFuncResults.csv'
print (f'Writing out to: {outputFileName}')
outputFile = open (outputFileName, 'w')
outputFile.write ('caseNum,expectedValue,predictedValue,state,index,df0,df1,df2,df3,df4,df5,df6,df7,df8,df9\n')

for c, case in enumerate (casesOfInterest):
    for index in case ['goodIndices']:
        decisionFuncResults = clf.decision_function(X_train[index].reshape(1, -1)).tolist()[0]

        outputFile.write (f"{c}, {case['expectedValue']}, {case['predictedValue']}, OK, {index}, ")
        for i in range (10):
            outputFile.write (str(decisionFuncResults[i]))
            if i != 9:
                outputFile.write (',')
            else:
                outputFile.write ('\n')

        fileName = f"FullClassifier_DecisionFuncResults/Case{c}_{case['expectedValue']}_{case['predictedValue']}_OK_{index}.jpg"
        digitImg = Image.fromarray (X_train[index].reshape (28, 28))
        ResultsImg = Image.new ('L', (28, 28), 255)
        ResultsImg.paste (digitImg, (0, 0))
        ResultsImg.save (fileName) 

    for index in case ['badIndices']:
        decisionFuncResults = clf.decision_function(X_train[index].reshape(1, -1)).tolist()[0]

        outputFile.write (f"{c}, {case['expectedValue']}, {case['predictedValue']}, KO, {index}, ")
        for i in range (10):
            outputFile.write (str(decisionFuncResults[i]))
            if i != 9:
                outputFile.write (',')
            else:
                outputFile.write ('\n')

        fileName = f"FullClassifier_DecisionFuncResults/Case{c}_{case['expectedValue']}_{case['predictedValue']}_KO_{index}.jpg"
        digitImg = Image.fromarray (X_train[index].reshape (28, 28))
        ResultsImg = Image.new ('L', (28, 28), 255)
        ResultsImg.paste (digitImg, (0, 0))
        ResultsImg.save (fileName) 

outputFile.close ()


