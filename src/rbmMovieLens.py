import pandas as pd
import numpy as np
import argparse
from rbm import RBM
import pdb

filePath = "/Users/me/Documents/SokratiMachineLearning/code/data/ml-100k/"
# Getting command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-u', '--hiddenUnits',type=int,help="Hidden units",dest='hiddenUnits')
parser.add_argument('-e', '--epochs',type=int,help="Epochs",dest='epochs')
parser.add_argument('-train', '--trainFile',help="File containing raw train data",dest='trainFile')
parser.add_argument('-test', '--testFile',help="File containing raw test data",dest='testFile')

args = parser.parse_args()

infoDataFrame = pd.read_csv(filePath+"u.info",sep=" ",header=None,usecols=[0,1])
infoMatrix = infoDataFrame.as_matrix()
totalUsers = infoMatrix[0][0]
totalMovies = infoMatrix[1][0]

# Data pre-processing using pandas
trainDataFrame = pd.read_csv(args.trainFile,delim_whitespace=True,header=None,usecols=[0,1,2])
trainDataFrame[2] = trainDataFrame[2].apply(lambda x: 1 if x>2 else 0)

testDataFrame = pd.read_csv(args.testFile,delim_whitespace=True,header=None,usecols=[0,1,2])
testDataFrame[2] = testDataFrame[2].apply(lambda x: 1 if x>2 else 0)

trainMatrix=trainDataFrame.as_matrix()
testMatrix=testDataFrame.as_matrix()

trainArray = np.ndarray(shape=(totalUsers,totalMovies), dtype=int)
row,column = trainMatrix.shape

# Converting raw dataframe into training numpy array
for i in range (0, row):
    userID = trainMatrix[i][0] -1
    movieID = trainMatrix[i][1] -1
    trainArray[userID][movieID]=trainMatrix[i][2]


r = RBM(num_visible = totalMovies,num_hidden = args.hiddenUnits)
r.train(trainArray, max_epochs = args.epochs)
print(r.weights)

testArray = np.zeros(shape=(1, totalMovies), dtype=int)
row,column = testMatrix.shape
print row, column
# Converting raw dataframe into testing numpy array
for i in range (0, row):
    userID = testMatrix[i][0] -1
    movieID = testMatrix[i][1] -1
    testArray[userID][movieID]=testMatrix[i][2]

print testArray[0].shape

movieDataFrame = pd.read_csv(filePath+"u.item",header=None,sep='|',usecols=[0,1])
print movieDataFrame
movieMatrix=movieDataFrame.as_matrix()
print movieMatrix
#pdb.set_trace()


tempArray = np.reshape(testArray[0], (1,totalMovies))
hidden_units = r.run_visible(tempArray)
print hidden_units
visible_units = r.run_hidden(hidden_units)
row, column = visible_units.shape
for v in range (0, column):
    if(visible_units[0][v]==1):
        print v, visible_units[0][v], movieMatrix[v][1]


