import numpy as np
from math import sqrt
import operator
import pandas as pd
from Classifiers.decision_tree import DecisionTree
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    a, b = np.array(a), np.array(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class RandomForest:
    def __init__(self, max_depth=5, max_features='sqrt', n_estimators=5, min_sample_leaf=0.1, feature_names=None):
        self.__test_features = None
        self.__test_labels = None
        self.__train_features = None
        self.__train_labels = None
        self.__predictions = None
        self.__feature_names = feature_names
        self.__max_depth = max_depth
        self.__max_features = max_features
        self.__n_estimators = n_estimators
        self.__min_sample_leaf = min_sample_leaf
        self.__trees = []
        print("Random Forest initiated.\n")

    def __feature_function(self):
        if isinstance(self.__max_features, int):
            assert(self.__max_features <= len(self.__train_features[0]))
            return self.__max_features

        if self.__max_features == "All":
            return len(self.__train_features[0])
        elif self.__max_features == "sqrt":
            return int(sqrt(len(self.__train_features[0])))
        else:
            return len(self.__train_features[0])

    def __build_forest(self, __=None):
        train_features, train_labels = unison_shuffled_copies(self.__train_features, self.__train_labels)
        random_val = np.random.randint(int(0.5 * len(train_features)), len(train_features) - 1)
        train_features = train_features[:random_val]
        train_labels = train_labels[:random_val]
        dt = DecisionTree(max_depth=self.__max_depth, feature_names=self.__feature_names,
                          min_sample_leaf=self.__min_sample_leaf, max_features=self.__max_features)
        dt.fit(np.array(train_features), np.array(train_labels))
        return dt

    def __build_forest2(self, __=None):
        for _ in range(self.__n_estimators):
            train_features, train_labels = unison_shuffled_copies(self.__train_features, self.__train_labels)
            random_val = np.random.randint(int(0.5 * len(train_features)), len(train_features) - 1)
            train_features = train_features[:random_val]
            train_labels = train_labels[:random_val]
            dt = DecisionTree(max_depth=self.__max_depth, feature_names=self.__feature_names,
                              min_sample_leaf=self.__min_sample_leaf, max_features=self.__max_features)
            dt.fit(np.array(train_features), np.array(train_labels))
            self.__trees.append(dt)

    def __classify(self, row):
        result_dict = {}
        for tree in self.__trees:
            result = tree.predict([row])[0]
            if result in result_dict:
                result_dict[result] += 1
            else:
                result_dict[result] = 1
        return max(result_dict.items(), key=operator.itemgetter(1))[0]

    def fit(self, train_feats, train_labels, feat_names=None):
        if self.__feature_names is None:
            self.__feature_names = feat_names
        self.__train_features, self.__train_labels = train_feats, train_labels
        #self.__build_forest2()
        pool = Pool(cpu_count())
        results = pool.amap(self.__build_forest, range(self.__n_estimators))
        self.__trees = results.get()

    def predict(self, test_feats):
        if self.__train_features is None or self.__train_labels is None:
            raise Exception("Training Data not fitted.")

        self.__test_features = test_feats
        self.__predictions = [self.__classify(row) for row in test_feats]
        return self.__predictions

    # Score model
    def score(self, test_labels):
        if self.__train_features is None or self.__train_labels is None:
            raise Exception("Training Data not fitted.")
        if self.__test_features is None:
            raise Exception("Test data not inputted.")

        self.__test_labels = test_labels
        for tree in self.__trees:
            tree.predict(self.__test_features)
            tree.score(test_labels, print_output=False)

        count = 0
        for prediction, result in zip(self.__predictions, self.__test_labels):
            if prediction == result:
                count += 1
        confusion_matrix = pd.crosstab(pd.Series(self.__predictions, name="Predicted"),
                                       pd.Series(self.__test_labels, name="Actual"))
        print("Accuracy: {0:.2f}%".format(count / len(self.__test_labels) * 100))
        print(confusion_matrix, "\n")

    def feature_importance(self):
        df_comb = self.__trees[0].feature_importance()
        for tree in self.__trees[1:]:
            df_comb = pd.concat([df_comb, tree.feature_importance()])
        return df_comb.groupby('Features')['Importance'].mean().sort_values(ascending=True)

    def print_tree(self, index):
        assert(index - 1 <= self.__n_estimators)
        self.__trees[index].print_tree()
