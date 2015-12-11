import utils
import numpy as np
import pandas as pd


def getData(size=None):
    train = pd.read_csv('./data/train.csv')
    train=train.fillna(0)
    # train=train.fillna(train.mean())
    train=train.values
    if size:
        train=train[:size,:]
    indices=list(range(1,train.shape[1]-1))#All indices except ID and except response
    # print(indices)
    changed=utils.categoricalToNumerical(train[:,2])
    print("number of categories of column 2:",changed[1])
    train[:,2]=changed[0]

    X, y = train[:,indices] , train[:,-1]

    y=list(map(np.int32,y))
    return X,y

# print(getData(size=10))
