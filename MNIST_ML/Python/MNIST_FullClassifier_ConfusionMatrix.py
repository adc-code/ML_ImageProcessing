#
# MNIST_FullClassifier_ConfusionMatrix.py
#
# Small utility to dump out the confusion matrix (CM) for the MNIST dataset when
# classified using stochastic gradient descent.  Also, the CM is dumped out in the
# form of a heat map as well.
#


import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


# 
# load the dataset
#
tic = time.perf_counter ()
mnist = fetch_openml ('mnist_784', version=1)
toc = time.perf_counter ()
loadTime = toc - tic
print (f'Time to preform Fetch: {loadTime:0.4f} seconds')


#
# create test and train sets...
#
TRAIN_TEST_SPLIT = 10000
X = mnist ['data']
y = mnist ['target']
y = y.astype (np.uint8)

X_train, X_test = X[:(X.shape[0] - TRAIN_TEST_SPLIT)], X[(X.shape[0] - TRAIN_TEST_SPLIT):]
y_train, y_test = y[:(X.shape[0] - TRAIN_TEST_SPLIT)], y[(X.shape[0] - TRAIN_TEST_SPLIT):]


#
# Make and train classifier
#
clf = SGDClassifier (random_state=42)

tic = time.perf_counter()
clf.fit (X_train, y_train)
toc = time.perf_counter()
trainTime = toc - tic
print (f'Time to preform fit data: {trainTime:0.4f} seconds')


#
# Determine the confusion matrix...
#
y_train_pred = cross_val_predict (clf, X_train, y_train, cv=3)
conf_mx = confusion_matrix (y_train, y_train_pred)

print ('Confusion matrix...')
print (conf_mx)


#
# Make a heat map of the confusion matrix...
#
plt.figure (figsize = (20,8))
sns.heatmap (conf_mx, annot=True, fmt="d", linewidths=.5, cmap="coolwarm", cbar=False, square=True)
plt.xlabel ('Predicted Results')
plt.ylabel ('Actual/Expected Values')
plt.savefig ('MNIST_FullClassifier_ConfusionMatrix.png')


#
# Make a heat map of the percent each error is of the whole so to determine which errors are
# perhaps the most 'interesting'.  Note that the diagonal is zeroed to avoid any distractions.
#
row_sums = conf_mx.sum (axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums * 100
np.fill_diagonal (norm_conf_mx, 0)
plt.figure (figsize = (20,8))
sns.heatmap (norm_conf_mx, linewidths=.5, cmap="coolwarm", square=True)
plt.xlabel ('Predicted Results')
plt.ylabel ('Actual/Expected Values')
plt.savefig ('MNIST_FullClassifier_CasesOfInterest.png')



