import pandas as pd
import numpy as np
import argparse
from rbm import RBM
import pdb

# Getting command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-u', '--hiddenUnits',type=int,help="Hidden units",dest='hiddenUnits')
parser.add_argument('-e', '--epochs',type=int,help="Epochs",dest='epochs')
parser.add_argument('-train', '--trainFile',help="File containing raw train data",dest='trainFile')
parser.add_argument('-test', '--testFile',help="File containing raw test data",dest='testFile')

args = parser.parse_args()

# Data pre-processing using pandas
trainDataFrame = pd.read_csv(args.trainFile,delim_whitespace=True,header=None,usecols=[0,1,2])
trainDataFrame[2] = trainDataFrame[2].apply(lambda x: 1 if x>2 else 0)

testDataFrame = pd.read_csv(args.testFile,delim_whitespace=True,header=None,usecols=[0,1,2])
testDataFrame[2] = testDataFrame[2].apply(lambda x: 1 if x>2 else 0)

trainMatrix=trainDataFrame.as_matrix()
testMatrix=testDataFrame.as_matrix()

trainArray = np.ndarray(shape=(943,1682), dtype=int)
row,column = trainMatrix.shape

# Converting raw dataframe into training numpy array
for i in range (0, row):
    userID = trainMatrix[i][0] -1
        movieID = trainMatrix[i][1] -1
        trainArray[userID][movieID]=trainMatrix[i][2]

totalUsers, totalMovies = trainArray.shape
r = RBM(num_visible = totalMovies,num_hidden = args.hiddenUnits)
r.train(trainArray, max_epochs = args.epochs)
print(r.weights)

testArray = np.zeros(shape=(943,1682), dtype=int)
row,column = testMatrix.shape
# Converting raw dataframe into training numpy array
for i in range (0, row):
    userID = testMatrix[i][0] -1
        movieID = testMatrix[i][1] -1
        testArray[userID][movieID]=testMatrix[i][2]

totalUsers, totalMovies = testArray.shape
print testArray[0].shape
for user in range(0, totalUsers-1):
    tempArray = np.reshape(testArray[user], (1,1682))
        print(r.run_visible(tempArray))	
