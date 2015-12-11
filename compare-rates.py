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
from dataAnalysis import getData


X,y=getData()
print("Data loaded!")
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,
random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean() )
clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
min_samples_split=1, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())
clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None,
min_samples_split=1, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())
exit(0)
clf = GradientBoostingClassifier(n_estimators=350, max_depth=None,
learning_rate=1, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())