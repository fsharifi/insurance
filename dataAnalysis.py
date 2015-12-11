import utils
import numpy as np
import pandas as pd


def getTrainData(size=None,featureIndices=None):
    train = pd.read_csv('./data/train.csv')
    train=train.fillna(0)
    # train=train.fillna(train.mean())
    train=train.values
    if size:
        train=train[:size,:]
    if featureIndices:
        indices=featureIndices
    else:
        indices=list(range(1,train.shape[1]-1))#All indices except ID and except response
    # print(indices)
    changed=utils.categoricalToNumerical(train[:,2])
    dict=changed[2]
    print("number of categories of column 2 of train:",changed[1])
    train[:,2]=changed[0]

    X, y = train[:,indices] , train[:,-1]

    y=list(map(np.int32,y))
    return X,y,dict


def getTestData(dict,size=None,featureIndices=None,):
    test = pd.read_csv('./data/test.csv')
    test=test.fillna(0)
    # train=train.fillna(train.mean())
    test=test.values
    if size:
        test=test[:size,:]
    if featureIndices:
        indices=featureIndices
    else:
        indices=list(range(1,test.shape[1]))#All indices except ID and except response
    # print(indices)
    changed=utils.categoricalToNumerical(test[:,2],dict)
    print("number of categories of column 2 of train and test:",changed[1])
    test[:,2]=changed[0]

    X= test[:,indices]
    ids=test[:,0]
    return X,ids


def saveToFile(predictions,ids,filename):

    submission = pd.DataFrame({
    "Id": ids,
    "Response": predictions
    })
    submission.to_csv(filename, index=False)

# Example
#
# X,y,dict=getTrainData(size=1000)
# print(X)
# print(y)
# testX,ids=getTestData(dict,size=1000)
# print(testX)
# print(ids)
# saveToFile(y,ids,"testSubmission.csv")
