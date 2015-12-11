__author__ = 'DeyerliQiz'

from dataAnalysis import getData
from sklearn import svm

print(svm)

X,y=getData()
clfs=\
    [svm.SVC()]
scores = cross_val_score(clf, X, y)
print(scores.mean() )
