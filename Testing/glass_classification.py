from Classifiers import k_nearest_neighbours, decision_tree, random_forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    glass_df = pd.read_csv("../Datasets/glass.csv")
    print(glass_df.columns)

    features = preprocessing.scale(glass_df.drop(["Type", "Fe"], axis=1).values)
    target = glass_df["Type"].values
    feature_names = glass_df.drop(["Type", "Fe"], axis=1).columns

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=69)

    knn = k_nearest_neighbours.KNearestNeighbours(k=100)
    knn.fit(X_train, y_train)
    knn.predict(X_test)
    knn.score(y_test)

    dt = decision_tree.DecisionTree(max_depth=15, max_features="sqrt", feature_names=feature_names, min_sample_leaf=0.1)
    dt.fit(X_train, y_train, feat_names=feature_names)
    dt.predict(X_test)
    dt.score(y_test)
    dt.print_tree()
    df = dt.feature_importance()
    print(df.head())
    df.plot.barh(x='Features', y='Importance')
    plt.show()

    rf = random_forest.RandomForest(max_depth=15, max_features="sqrt", feature_names=feature_names,
                                min_sample_leaf=0.1, n_estimators=500)
    rf.fit(X_train, y_train)
    rf.predict(X_test)
    rf.score(y_test)
    df = rf.feature_importance()
    print(df.head())
    df.plot.barh(x='Features', y='Importance')
    plt.show()

    from sklearn.metrics import precision_recall_fscore_support

    rf = RandomForestClassifier(max_depth=15, n_estimators=5000, max_features="auto", criterion='entropy')
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(precision_recall_fscore_support(y_test, predictions))


