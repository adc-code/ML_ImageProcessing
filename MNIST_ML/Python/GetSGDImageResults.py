#
# GetSGDImageResults.py: utility script used to extract key details for the image results 
#


import pandas as pd

# read the data file...
df = pd.read_csv ('BinaryClassifier_TotalResults.csv')

# filter out the SGD data and drop unneeded columns
dfSGD = df [df ['ClassiferType'] == 'SGD']
dfSGD.drop ( ['ClassiferType', 'loadTime', 'trainTime', 'rocMetricTime', \
              'meanPredTime', 'precision', 'recall', 'F1Score'], axis=1, inplace=True )

# Make a new column of the total...
dfSGD ['total'] = dfSGD ['numTruePos'] + dfSGD ['numTrueNeg'] + dfSGD ['numFalsePos'] + dfSGD ['numFalseNeg']

# write it to the screen
print ( dfSGD )

