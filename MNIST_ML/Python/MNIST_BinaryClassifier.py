#
# MNIST_BinaryClassifier.py
#
# Script used to classify one number from the MNIST dataset.  The results simply
# evaluate if a number is Number / not Number or not a number
#
# Usage: > python MNIST_BinaryClassifier.py  <Number>  <ClassifierType>  <OutputFile>
#
# Where ClassifierType is:
#     1 --> Stochastic Gradient Descent
#     2 --> Random Forest Classifier
#     3 --> Logistic Regression 
#     4 --> KNeighborsClassifier
#


# for command line args...
import sys

# used to output images from resulting classification
import random
from PIL import Image

# for performance monitoring
import time
import math

# for interaction with the data
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# classifier metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def PrintClassifierMsg ():
    print ('Classifier Type:')
    print ('   1 --> Stochastic Gradient Descent')
    print ('   2 --> Random Forest Classifier')
    print ('   3 --> Logistic Regression') 
    print ('   4 --> KNeighborsClassifier') 


if len (sys.argv) != 3:
    print ('Error: incorrect number of parameters!')
    print ()
    print ('Usage:', sys.argv[0],'  <Number>  <ClassifierType>  ')
    print ()
    PrintClassifierMsg ()
    sys.exit ()
else:
    if int (sys.argv[2]) not in [ 1, 2, 3, 4 ]:
        print ('Incorrect classifier number')
        PrintClassifierMsg ()
        sys.exit ()


Number = int (sys.argv [1])
RegressionType = int (sys.argv [2]) - 1
RegressionTypeStr = [ 'SGD', 'RandomForest', 'Logistic', 'KNeighbors' ]

TRAIN_TEST_SPLIT = 10000
IMG_WIDTH  = 28
IMG_HEIGHT = 28
IMG_COUNT  =  6


print ('#')
print ('# Classifier: ', RegressionTypeStr [RegressionType])
print ('#')
print ()

print ('#')
print ('# Number: ', Number)
print ('#')
print ()



#
# Get the data...
#
tic = time.perf_counter ()
mnist = fetch_openml ('mnist_784', version=1)
toc = time.perf_counter ()
loadTime = toc - tic
print (f'Time to preform Fetch: {loadTime:0.4f} seconds')
print ()


#
# Establish X and y...
#
X = mnist ['data']
y = mnist ['target']
y = y.astype (np.uint8)


#
# Create test and train sets...
#
X_train, X_test = X[:(X.shape[0] - TRAIN_TEST_SPLIT)], X[(X.shape[0] - TRAIN_TEST_SPLIT):]
y_train, y_test = y[:(X.shape[0] - TRAIN_TEST_SPLIT)], y[(X.shape[0] - TRAIN_TEST_SPLIT):]
y_train_N = (y_train == Number) 
y_test_N  = (y_test  == Number)


#
# Make the appropriate model...
#
if RegressionType == 0: 
    clf = SGDClassifier (random_state=42)
elif RegressionType == 1:
    clf = RandomForestClassifier (random_state=42, n_estimators=50)
elif RegressionType == 2:
    clf = LogisticRegression (solver='lbfgs', max_iter=1000, random_state=42)
elif RegressionType == 3:
    clf = KNeighborsClassifier (weights='distance', n_neighbors=4)


#
# Train the data...
#
tic = time.perf_counter()
clf.fit (X_train, y_train_N)
toc = time.perf_counter()
trainTime = toc - tic
print (f'Time to preform fit data: {trainTime:0.4f} seconds')


#
# Compute various classifier metrics...
#
tic = time.perf_counter()
if RegressionType == 0 or RegressionType == 2:
    y_scores = cross_val_predict (clf, X_train, y_train_N, cv=3, method="decision_function")
    FPR, TPR, thresholds = roc_curve (y_train_N, y_scores)
    AUC = roc_auc_score (y_train_N, y_scores)
elif RegressionType == 1 or RegressionType == 3:
    y_proba = clf.predict_proba (X_test)[:, 1]
    FPR, TPR, thresholds = roc_curve (y_test_N, y_proba)
    AUC = roc_auc_score (y_test_N, y_proba)
toc = time.perf_counter()
rocMetricTime = toc - tic
print (f'Time to preform cross val predict function: {rocMetricTime:0.4f} seconds')


#
# Evaluate test set...
#
TruePos  = []
TrueNeg  = []
FalsePos = []
FalseNeg = []
prevPctDone = -1
predTimes = []

prevTime = time.perf_counter()
for i in range (len(X_test)):
    predTest = clf.predict (X_test[i].reshape(1, -1))

    pctDone = round (math.floor (round (100 * i / len(X_test)) / 1) * 1)
    if pctDone != prevPctDone:
        currTime = time.perf_counter()
        print (f'Predictions done... {pctDone}%    ({(currTime - prevTime):0.4f}s)')
        predTimes.append (currTime - prevTime);
        prevPctDone = pctDone
        prevTime = currTime

    if predTest == True and y_test_N [i] == True:
        TruePos.append (i)
    elif predTest == False and y_test_N [i] == False:
        TrueNeg.append (i)
    elif predTest == True and y_test_N [i] == False:
        FalsePos.append (i)
    elif predTest == False and y_test_N [i] == True:
        FalseNeg.append (i)

print ()

meanPredTime = 0
for predTime in predTimes:
    meanPredTime += predTime
meanPredTime /= len(predTimes)


#
# Create the images of the various results... so we can see what some of the
# false positive & negative digits were
#
for k in range (4):
    ResultsImg = Image.new ('L', (IMG_COUNT * IMG_WIDTH, IMG_COUNT * IMG_HEIGHT), 255)

    # make a sample of the results...
    index = 0
    for i in range (IMG_COUNT):
        for j in range (IMG_COUNT):

            digit = []
            if k == 0:
                if index < len(TruePos):
                    digit = X_test[TruePos[index]]
            elif k == 1:
                if index < len(TrueNeg):
                    digit = X_test[TrueNeg[index]]
            elif k == 2:
                if index < len(FalsePos):
                    digit = X_test[FalsePos[index]]
            elif k == 3:
                if index < len(FalseNeg):
                    digit = X_test[FalseNeg[index]]

            if len(digit) > 0: 
                 digitImg = Image.fromarray (digit.reshape (IMG_WIDTH, IMG_HEIGHT))
                 ResultsImg.paste (digitImg, (i * IMG_WIDTH, j * IMG_HEIGHT))

            index += 1

    # finally, write out the image
    fileName = 'BinaryClassifier_' + RegressionTypeStr[RegressionType] + '_' +  str(Number) + '_'
    if k == 0:
        fileName += 'TP'
    elif k == 1:
        fileName += 'TN'
    elif k == 2:
        fileName += 'FP'
    elif k == 3:
        fileName += 'FN'
    fileName += '.jpg'
    
    print (f'Writing out to: {fileName}')
    ResultsImg.save (fileName)


#
# write out key results to a CSV file
#
fileName = 'BinaryClassifier_Results_' + RegressionTypeStr[RegressionType] + '_' +  str(Number) + '.csv'
print (f'Writing out to: {fileName}')
resFile = open (fileName, 'w')
   
resFile.write (RegressionTypeStr [RegressionType] + ', ')
resFile.write (str(Number) + ', ')

resFile.write (str(loadTime) + ', ')
resFile.write (str(trainTime) + ', ')
resFile.write (str(rocMetricTime) + ', ')
resFile.write (str(meanPredTime) + ', ')

resFile.write (str(len(TruePos))  + ', ')
resFile.write (str(len(TrueNeg))  + ', ')
resFile.write (str(len(FalsePos)) + ', ')
resFile.write (str(len(FalseNeg)) + ', ')

resFile.write (str(len(TruePos) / (len(TruePos) + len(FalsePos))) + ', ')
resFile.write (str(len(TruePos) / (len(TruePos) + len(FalseNeg))) + ', ')
resFile.write (str(len(TruePos) / (len(TruePos) + (len(FalseNeg) + len(FalsePos)) / 2)) + '\n')

resFile.close ()


#
# write out ROC curve data to a Json file
#
fileName = 'BinaryClassifier_Results_' + RegressionTypeStr[RegressionType] + '_' +  str(Number) + '.json'
print (f'Writing out to: {fileName}')
resFile = open (fileName, 'w')
   
resFile.write ('{\n')
resFile.write (f'   "RegType": "{RegressionTypeStr[RegressionType]}",\n')
resFile.write (f'   "Number": {Number},\n')
resFile.write (f'   "AUC": {AUC},\n')

resFile.write (f'   "FPR": [ ')
for i in range(len(FPR)):
    if i != len(FPR)-1:
        resFile.write (f'{FPR[i]}, ')
    else:
        resFile.write (f'{FPR[i]} ], \n')

resFile.write (f'   "TPR": [ ')
for i in range(len(TPR)):
    if i != len(TPR)-1:
        resFile.write (f'{TPR[i]}, ')
    else:
        resFile.write (f'{TPR[i]} ]\n')

resFile.write ('}\n')
resFile.close ()

print ()



