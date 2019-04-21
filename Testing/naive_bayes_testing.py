from sklearn import datasets
from sklearn.model_selection import train_test_split
from Classifiers.naive_bayes import NaiveBayes

# Iris Dataset test
iris = datasets.load_iris()
features = iris.data
target = [iris.target_names[int(target)] for target in iris.target]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=3)

nb = NaiveBayes()
nb.fit(X_train, y_train)
nb.predict(X_test, y_test)
nb.score()
#

# Digits Dataset Test
wine = datasets.load_wine()
features = wine.data
target = [wine.target_names[int(target)] for target in wine.target]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=3)

nb = NaiveBayes()
nb.fit(X_train, y_train)
nb.predict(X_test, y_test)
nb.score()
#

"""
Naive Bayes Initiated Using using gaussian distribution.

Accuracy: 96.00%
Actual      setosa  versicolor  virginica
Predicted                                
setosa          17           0          0
versicolor       0          15          1
virginica        0           1         16 

Naive Bayes Initiated Using using gaussian distribution.

Accuracy: 96.61%
Actual     class_0  class_1  class_2
Predicted                           
class_0         18        0        0
class_1          2       23        0
class_2          0        0       16 
"""
