import utils
import numpy as np
import pandas as pd


train = pd.read_csv('./data/train.csv')
# print(train.isnull().sum())
train=train.fillna(0)
# print(train.isnull().sum())

train=train.values
# train=train[0:100,:]
# Build a classification task using 3 informative features
indeces=(range(train.shape[1]-1))
indeces=list(set(indeces)-set([0]))#removing ID
# print(indeces)
# print(train.shape[1])
changed=utils.categoricalToNumerical(train[:,2])
print("number of categories of column 2:",changed[1])
train[:,2]=changed[0]

X, y = train[:,indeces] , train[:,-1]
y=list(map(np.int32,y))

print(type(y))
print(type(y[0]))
print(y)
numfeatures=len(indeces)
print(type(X))