import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from dataAnalysis import getTrainData

importance=[10,9,3,7,40,8,11,16,1,38,37,19,35,33,12,59,34,51,
92,28,36,32,77,22,45,65,5,17,14,70,30,27,20,80,52,72
,24,61,62,75,29,39,57,2,13,44,49,64,15,102,114,60,66,31
,54,26,48,73,100,69,88,117,25,0,87,119,78,99,42,43,76,55
,23,125,53,6,50,110,56,21,124,109,58,41,111,101,107,81,98,105
,84,103,18,63,93,122,67,108,83,113,106,116,4,115,104,82,120,89
,118,94,79,96,86,68,47,97,95,123,46,121,85,91,112,90,74,71]

Test = pd.read_csv('./data/test.csv')
X,y=getTrainData()
print("Data loaded!")
X=X[:,importance[1:30]]

clf = GradientBoostingClassifier(n_estimators=1000, max_depth=None,
learning_rate=1, random_state=0).fit(X,y)
ytest=clf.predict(Test)
print(ytest.shape)
submission = pd.DataFrame({
        "Id": Test["Id"],
        "Response": ytest
    })
submission.to_csv('XGBOOST.csv', index=False)

