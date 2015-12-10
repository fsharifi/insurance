print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


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

print(type(y))
print(type(y[0]))
print(y)
numfeatures=len(indeces)
print(type(X))
# X=X[:,indeces]
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier()

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()