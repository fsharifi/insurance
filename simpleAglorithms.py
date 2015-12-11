__author__ = 'DeyerliQiz'

from dataAnalysis import getData
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score

X,y=getData(size=10)
print("data loaded!")

clfs=\
    [svm.SVC()]

for clf in clfs:
    print(clf)
    scores = cross_val_score(clf, X, y)
    print(scores.mean())

