__author__ = 'DeyerliQiz'

from dataAnalysis import getData
from sklearn import svm

print(svm)

X,y=getData(size=10)
clfs=\
    []
scores = cross_val_score(clf, X, y)
print(scores.mean() )
