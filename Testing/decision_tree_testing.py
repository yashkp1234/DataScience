from sklearn import datasets
from sklearn.model_selection import train_test_split
from Classifiers.decision_tree import DecisionTree
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import time
    start = time.time()
    np.random.seed(60)

    dt = DecisionTree(max_depth=2, max_features="sqrt")
    iris = datasets.load_iris()
    features = iris.data
    target = [iris.target_names[int(target)] for target in iris.target]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=9)

    column_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

    dt.fit(X_train, y_train, column_names)
    dt.print_tree()
    dt.predict(X_test)
    dt.score(y_test)

    df = dt.feature_importance()
    print(df.head())
    df.plot.barh(x='Features', y='Importance')
    plt.show()

    # Long dictionary copied from website to rename attributes in features
    dict = "{'cap-shape': {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 'k': 'knobbed', 's': 'sunken'}, " \
           "'cap-surface': {'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'}, " \
           "'cap-color': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'r': 'green', 'p': 'pink', " \
                "'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'}, 'bruises': {'t': 'bruises', 'f': 'no'}, " \
           "'odor': {'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy', 'f': 'foul', 'm': 'musty', 'n': 'none', 'p': 'pungent', 's': 'spicy'}, " \
           "'gill-attachment': {'a': 'attached', 'd': 'descending', 'f': 'free', 'n': 'notched'}, " \
           "'gill-spacing': {'c': 'close', 'w': 'crowded', 'd': 'distant'}, " \
           "'gill-size': {'b': 'broad', 'n': 'narrow'}, " \
           "'gill-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray', 'r': 'green', 'o': 'orange', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'}, " \
           "'stalk-shape': {'e': 'enlarging', 't': 'tapering'}, " \
           "'stalk-root': {'b': 'bulbous', 'c': 'club', 'u': 'cup', 'e': 'equal', 'z': 'rhizomorphs', 'r': 'rooted', 'missing': '?'}, " \
           "'stalk-surface-above-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'}, " \
           "'stalk-surface-below-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'}, " \
           "'stalk-color-above-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'}, " \
           "'stalk-color-below-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'}, " \
           "'veil-type': {'p': 'partial', 'u': 'universal'}, 'veil-color': {'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'}, " \
           "'ring-number': {'n': 'none', 'o': 'one', 't': 'two'}, " \
           "'ring-type': {'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring', 'l': 'large', 'n': 'none', 'p': 'pendant', 's': 'sheathing', 'z': 'zone'}, " \
           "'spore-print-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'r': 'green', 'o': 'orange', 'u': 'purple', 'w': 'white', 'y': 'yellow'}, " \
           "'population': {'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered', 'v': 'several', 'y': 'solitary'}, " \
           "'habitat': {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'u': 'urban', 'w': 'waste', 'd': 'woods'}}"
    dict = ast.literal_eval(dict)

    dt = DecisionTree(max_depth=3)
    mushrooms = pd.read_csv("../Datasets/mushrooms.csv")
    target = mushrooms["class"]
    features = mushrooms.drop("class", axis=1)
    column_names = list(features.columns)

    # Rename feature attributes
    for col in column_names:
        def func(x):
            if x == "?":
                return "missing"
            else:
                return dict[col][x]
        features[col] = features[col].apply(lambda x: func(x))


    X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, test_size=0.33, random_state=9)

    dt.fit(X_train, y_train, column_names)
    dt.print_tree()
    dt.predict(X_test)
    dt.score(y_test)

    df = dt.feature_importance()
    print(df.head())
    df.plot.barh(x='Features', y='Importance')
    plt.show()
    print(time.time() - start)

"""
Results Below:
Decision Tree initiated.

Decision Tree: 
Question: Is Petal Length >= 3.3?
    True --> 
    Question: Is Petal Width >= 1.8?
        True --> {'versicolor': '3.12%', 'virginica': '96.88%'}
        False --> {'versicolor': '86.84%', 'virginica': '13.16%'}
    False --> {'setosa': '100.00%'}

Accuracy: 98.00%
Actual      setosa  versicolor  virginica
Predicted                                
setosa          20           1          0
versicolor       0          15          0
virginica        0           0         14 

Decision Tree initiated.

Decision Tree: 
Question: Is odor == none?
    True --> 
    Question: Is spore-print-color == green?
        True --> {'p': '100.00%'}
        False --> 
        Question: Is stalk-surface-below-ring == scaly?
            True --> {'p': '72.73%', 'e': '27.27%'}
            False --> {'e': '99.73%', 'p': '0.27%'}
    False --> 
    Question: Is stalk-root == club?
        True --> 
        Question: Is bruises == bruises?
            True --> {'e': '100.00%'}
            False --> {'p': '100.00%'}
        False --> 
        Question: Is stalk-root == rooted?
            True --> {'e': '100.00%'}
            False --> {'p': '97.52%', 'e': '2.48%'}

Accuracy: 98.51%
Actual        e     p
Predicted            
e          1376     2
p            38  1265 
"""
