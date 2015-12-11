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



train = pd.read_csv('./data/train.csv')
# print(train.isnull().sum())
train=train.fillna(0)
# print(train.isnull().sum())

train=train.values
#train=train[0:100,:]
# Build a classification task using 3 informative features
indeces=(range(train.shape[1]-1))
indeces=list(set(indeces)-set([0,2]))
# print(indeces)
# print(train.shape[1])
# exit(0)
X, y = train[:,indeces] , train[:,-1]
y=list(map(np.int32,y))

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,
random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean() )
clf = RandomForestClassifier(n_estimators=10, max_depth=None,
min_samples_split=1, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean() )
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
min_samples_split=1, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean() )


clf = GradientBoostingClassifier(n_estimators=10, max_depth=None,
learning_rate=1, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean() )