__author__ = 'DeyerliQiz'

from dataAnalysis import getTrainData
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier

X,y=getTrainData(size=10)
print("data loaded!")

clfs=\
    [svm.SVC(),
     # NearestNeighbors(n_neighbors=7, algorithm='ball_tree'),
     RandomForestClassifier(n_estimators=10)]

for clf in clfs:
    print(clf)
    scores = cross_val_score(clf, X, y)#,scoring="accuracy"
    print(scores.mean())

