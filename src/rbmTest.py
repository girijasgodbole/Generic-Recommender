import cPickle as pickle
import numpy as np
import pandas as pd
import argparse

filePath = "/Users/me/Documents/SokratiMachineLearning/code/data/ml-100k/"

# Getting command line arguments
parserTest = argparse.ArgumentParser(description='Process testing specifications')

parserTest.add_argument('-model', '--modelFile',help="select the model",dest='model')
parserTest.add_argument('-test', '--testFile',help="File containing raw test data",dest='testFile')
# parser.add_argument('-train', '--trainFile',help="File containing raw train data",dest='trainFile')

args = parserTest.parse_args()

# unpickling
fp = open(args.model, "rb")
infoMatrix = pickle.load(fp)
movieMatrix = pickle.load(fp)
rbmObject = pickle.load(fp)

totalMovies = infoMatrix[1][0]

testDataFrame = pd.read_csv(args.testFile,delim_whitespace=True,header=None,usecols=[0,1,2])
testDataFrame[2] = testDataFrame[2].apply(lambda x: 1 if x>2 else 0)
testMatrix=testDataFrame.as_matrix()
testArray = np.zeros(shape=(1, totalMovies), dtype=int)

row,column = testMatrix.shape
print row, column

# Converting raw dataframe into testing numpy array
for i in range (0, row):
    userID = testMatrix[i][0] -1
    movieID = testMatrix[i][1] -1
    testArray[userID][movieID]=testMatrix[i][2]

print testArray[0].shape

tempArray = np.reshape(testArray[0], (1,totalMovies))
hidden_units = rbmObject.run_visible(tempArray)
print hidden_units
visible_units = rbmObject.run_hidden(hidden_units)
row, column = visible_units.shape
for v in range (0, column):
    if(visible_units[0][v]==1):
        print v, visible_units[0][v], movieMatrix[v][1]