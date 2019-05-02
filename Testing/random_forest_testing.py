from sklearn import datasets
from sklearn.model_selection import train_test_split
from Classifiers.random_forest import RandomForest
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    np.random.seed(10)

    iris = datasets.load_iris()
    features = iris.data
    target = [iris.target_names[int(target)] for target in iris.target]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=9)

    rf = RandomForest(max_depth=2, max_features='All', n_estimators=10, min_sample_leaf=0.25)
    rf.fit(X_train, y_train, ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
    rf.predict(X_test)
    rf.score(y_test)
    df = rf.feature_importance()
    print(df)
    df.plot.barh(x='Features', y='Importance')
    plt.show()
    rf.print_tree(3)


"""
Results Below:

Random Forest initiated.

Accuracy: 90.00%
Actual      setosa  versicolor  virginica
Predicted                                
setosa          20           4          0
versicolor       0          12          1
virginica        0           0         13 

Features
Sepal Length     6.005
Sepal Width      8.295
Petal Length    25.175
Petal Width     60.525
Name: Importance, dtype: float64

Decision Tree: 
Question: Is Petal Length >= 5.1?
    True --> {'virginica': '100.00%'}
    False --> 
    Question: Is Sepal Length >= 5.6?
        True --> {'versicolor': '100.00%'}
        False --> {'setosa': '100.00%'}
"""