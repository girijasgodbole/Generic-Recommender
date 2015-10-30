import pandas as pd
import numpy as np
import argparse
from rbm import RBM
import pdb
import cPickle as pickle

filePath = "/Users/me/Documents/SokratiMachineLearning/code/data/ml-100k/"

# load dataset specifications
infoDataFrame = pd.read_csv(filePath+"u.info",sep=" ",header=None,usecols=[0,1])
infoMatrix = infoDataFrame.as_matrix()
totalUsers = infoMatrix[0][0]
totalMovies = infoMatrix[1][0]

# load movieId-movieName specifications
movieDataFrame = pd.read_csv(filePath+"u.item",header=None,sep='|',usecols=[0,1])
print movieDataFrame
movieMatrix=movieDataFrame.as_matrix()
print movieMatrix

# Getting command line arguments
parser = argparse.ArgumentParser(description='Process training specifications')

parser.add_argument('-u', '--hiddenUnits',type=int,help="Hidden units",dest='hiddenUnits')
parser.add_argument('-e', '--epochs',type=int,help="Epochs",dest='epochs')
parser.add_argument('-train', '--trainFile',help="File containing raw train data",dest='trainFile')

args = parser.parse_args()

# Data pre-processing using pandas
trainDataFrame = pd.read_csv(args.trainFile,delim_whitespace=True,header=None,usecols=[0,1,2])
trainDataFrame[2] = trainDataFrame[2].apply(lambda x: 1 if x>2 else 0)
trainMatrix=trainDataFrame.as_matrix()
trainArray = np.ndarray(shape=(totalUsers,totalMovies), dtype=int)
row,column = trainMatrix.shape

# Converting raw dataframe into training numpy array
for i in range (0, row):
    userID = trainMatrix[i][0] -1
    movieID = trainMatrix[i][1] -1
    trainArray[userID][movieID]=trainMatrix[i][2]

# training
rbmObject = RBM(num_visible = totalMovies,num_hidden = args.hiddenUnits)
rbmObject.train(trainArray, max_epochs = args.epochs)
print(rbmObject.weights)

# pickling
modelName = "model-"+str(args.epochs)+"-"+str(args.hiddenUnits)+".p"
print modelName
fp = open( modelName, "wb" )
for obj in [infoMatrix, movieMatrix, rbmObject]:
    pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)

print 'Model trained and pickled'


