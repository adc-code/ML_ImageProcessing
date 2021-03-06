#
# MNIST_BinaryClassifier_BuildResults.py
#
# Helper script used to try out different combinations of number and classifiers.  Also
# this script merges the results together into two files: one csv for various metrics,
# and one json file ROC Curve data.
#
# Usage:
# > python MNIST_BinaryClassifier_BuildResults.py  <Tasks>
# 
# Where <Tasks> is:
#    - GENERATE_TESTS : to just run all the test cases
#    - MERGE_RESULTS : to just merge the results from the test cases
#    - GENERATE_AND_MERGE : do both running and merging 
#


import os
import sys


if len (sys.argv) != 2:

    print ('Error: incorrect command line parameters...')
    print ()
    print (f'Usage: {sys.argv[0]}  <Tasks>')
    print ()
    print ('Where <Tasks> is: ')
    print ('  --> GENERATE_TESTS ')
    print ('  --> MERGE_RESULTS ')
    print ('  --> GENERATE_AND_MERGE ')
    sys.exit ()


Tasks = 3
if sys.argv[1] == 'GENERATE_TESTS':
    Tasks = 1
elif sys.argv[1] == 'MERGE_RESULTS':
    Tasks = 2


print ('Tasks: ', Tasks)
possibleNumbers = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
possibleClassifiers = [ 1, 2, 3, 4 ]


#
# Generating results
#
if Tasks == 1 or Tasks == 3:

    script = 'MNIST_BinaryClassifier.py'

    for classifier in possibleClassifiers:
        for number in possibleNumbers:
            toExecute = script + ' ' + str(number) + ' ' + str(classifier)

            print ('>>')
            print ('>> Excuting... ', toExecute)
            print ('>>')
            print ()

            toExecute = 'python ' + toExecute
            os.system (toExecute)


#
# merging results
#
if Tasks == 2 or Tasks == 3:

    os.system ('touch BinaryClassifier_TotalResults.csv')
    os.system ('rm BinaryClassifier_TotalResults.csv')
    os.system ('touch BinaryClassifier_TotalROCCurvess.json')
    os.system ('rm BinaryClassifier_TotalROCCurvess.json')

    os.system ("echo '[' >> BinaryClassifier_TotalROCCurvess.json")

    for classifier in [ 'SGD', 'RandomForest', 'Logistic', 'KNeighbors' ]:
        for i in range(len(possibleNumbers)):
            resultsFile = 'BinaryClassifier_Results_' + classifier + '_' + str(possibleNumbers[i]) + '.csv'
            ROCCurveFile = 'BinaryClassifier_Results_' + classifier + '_' + str(possibleNumbers[i]) + '.json'

            os.system ('cat ' + resultsFile + ' >> BinaryClassifier_TotalResults.csv')
            os.system ('cat ' + ROCCurveFile + ' >> BinaryClassifier_TotalROCCurvess.json')

            if i != len(possibleNumbers)-1:
                os.system ("echo ', ' >> BinaryClassifier_TotalROCCurvess.json")

    os.system ("echo ']' >> BinaryClassifier_TotalROCCurvess.json")

    print ('Writen to... BinaryClassifier_TotalResults.csv')
    print ('Writen to... BinaryClassifier_TotalROCCurvess.json')


