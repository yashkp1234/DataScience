from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNearestNeighbours.k_nearest_neighbours import KNearestNeighbours

# Iris Dataset test
iris = datasets.load_iris()
features = iris.data
target = [iris.target_names[int(target)] for target in iris.target]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=3, stratify=target)

knn = KNearestNeighbours(k=5)
knn.fit(X_train, y_train)
knn.predict(X_test, y_test)
knn.score()
#

# Digits Dataset Test
digits = datasets.load_digits()
features = digits.data
target = digits.target

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=3, stratify=target)

knn = KNearestNeighbours(k=5, calc_method="manhattan")
knn.fit(X_train, y_train)
knn.predict(X_test, y_test)
knn.score()
#

'''
Results below if unable to run:

KNN Initiated Using k=5 using euclidean distance.

Accuracy: 96.00%
Actual      setosa  versicolor  virginica
Predicted                                
setosa          17           0          0
versicolor       0          14          0
virginica        0           2         17

KNN Initiated Using k=5 using manhattan distance.

Accuracy: 98.48%
Actual      0   1   2   3   4   5   6   7   8   9
Predicted                                        
0          59   0   0   0   0   0   0   0   0   0
1           0  60   0   0   0   0   0   0   3   0
2           0   0  59   0   0   0   0   0   1   0
3           0   0   0  60   0   0   0   0   0   0
4           0   0   0   0  60   0   0   0   0   0
5           0   0   0   0   0  58   0   0   0   2
6           0   0   0   0   0   0  60   0   0   0
7           0   0   0   0   0   0   0  58   0   0
8           0   0   0   0   0   0   0   0  54   0
9           0   0   0   0   0   2   0   1   0  57

'''

