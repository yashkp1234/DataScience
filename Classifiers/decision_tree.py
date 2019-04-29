import operator
import pandas as pd
import numpy as np
from re import match as re_match


# Get the count of each class in data
def class_counts(rows):
    counts = {}
    for row in rows:
        # the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# Partition the data into two sets, one where rows return true to question and one where they return false
def partition_function(data, question):
    data_true, data_false = [], []
    for row in data:
        if question.match(row):
            data_true.append(row)
        else:
            data_false.append(row)
    return data_true, data_false


# Calculate Gini index for the data
def gini_index(data):
    if len(data) < 0:
        raise Exception("Data has no rows")

    class_count = class_counts(data)
    impurity = 1
    for key in class_count:
        impurity -= (class_count[key] / len(data)) ** 2
    return impurity


# Calculate the information gain by splitting the data
def information_gain(data_left, data_right, uncertainty):
    if len(data_left) + len(data_right) == 0:
        raise Exception("Data has no rows")

    p = len(data_left) * 1.0 / (len(data_right) + len(data_left))
    return uncertainty - p * gini_index(data_left) - (1 - p) * gini_index(data_right)


# Check if a value is number / could be interpreted as a number
def is_numeric(val):
    if isinstance(val, int) or isinstance(val, float):
        return True
    else:
        if re_match("^\d+?\.\d+?$", val) is None:
            return val.isdigit()
        return True


class DecisionTree:

    # Constructor
    def __init__(self, max_depth=None, feature_names=None):
        self.__test_features = None
        self.__test_labels = None
        self.__train_features = None
        self.__train_data = None
        self.__train_labels = None
        self.__predictions = None
        self.__feature_names = feature_names
        self.__max_depth = max_depth
        self.__tree = None
        print("Decision Tree initiated.\n")

    # Define a leaf class to contain final prediction values
    class Leaf:
        def __init__(self, data):
            class_count = class_counts(data)
            sum_vals = sum(class_count.values())
            self.predictions = class_count
            for key in self.predictions:
                self.predictions[key] = "{0:.2f}%".format(self.predictions[key]/sum_vals * 100)

        # Overload string conversion function to be used for printing
        def __str__(self):
            return str(self.predictions)

        # Classify function to classify an input row
        def classify(self, row=None):
            return max(self.predictions.items(), key=operator.itemgetter(1))[0]

    # Decision node class holds question, and true and false branches
    class DecisionNode:
        # Constructor
        def __init__(self, question, true_branch, false_branch, depth):
            self.question, self.true_branch, self.false_branch, self.depth = question, true_branch, false_branch, depth

        # Overload string conversion function to be used for printing
        def __str__(self):
            spacing = ""
            if self.depth != 0:
                spacing = "\n" + self.depth * "    "
            spacing2 = (self.depth + 1) * "    "
            return "%sQuestion: %s\n%sTrue --> %s\n%sFalse --> %s" % \
                   (spacing, str(self.question), spacing2, str(self.true_branch), spacing2, str(self.false_branch))

        # Classify function to classify an input row
        def classify(self, row):
            if self.question.match(row):
                return self.true_branch.classify(row)
            else:
                return self.false_branch.classify(row)

    # Question class used to store data splitting information
    class Question:
        # Constructor
        def __init__(self, column, value, feature_names=None):
            self.index, self.value, self.feature_names = column, value, feature_names

        # Return true if row answers the question
        def match(self, row):
            val = row[self.index]
            if is_numeric(val):
                return float(val) >= float(self.value)
            else:
                return str(val) == str(self.value)

        # Overload string conversion function to be used for printing
        def __str__(self):
            condition = "=="
            if is_numeric(self.value):
                condition = ">="
            if self.feature_names is None:
                return "Is feature " + str(self.index) + " " + condition + " " + str(self.value) + "?"
            else:
                return "Is " + str(self.feature_names[self.index]) + " " + condition + " " + str(self.value) + "?"

    # Iterate over features and find best split using Gini Index
    def __find_best_split(self, data):
        if len(data) == 0:
            raise Exception("Data has no rows")

        uncertainty = gini_index(data)
        best_gain = 0
        best_question = None
        num_feats = len(data[0]) - 1

        for index in range(num_feats):
            possible_values = set([row[index] for row in data])

            for val in possible_values:
                question = self.Question(index, val, self.__feature_names)
                true_rows, false_rows = partition_function(data, question)

                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                gain = information_gain(data_left=true_rows, data_right=false_rows, uncertainty=uncertainty)

                if not best_gain or best_gain < gain:
                    best_question, best_gain = question, gain

        return best_question, best_gain

    # Build the decision tree
    def __build_tree(self, data, depth):
        question, gain = self.__find_best_split(data)

        if gain == 0 or not gain or (self.__max_depth and depth == self.__max_depth):
            return self.Leaf(data)

        true_rows, false_rows = partition_function(data, question)

        true_branch = self.__build_tree(true_rows, depth + 1)
        false_branch = self.__build_tree(false_rows, depth + 1)

        return self.DecisionNode(question, true_branch, false_branch, depth)

    # Fit data
    def fit(self, train_feats, train_labels, feat_names=None):
        self.__train_features, self.__train_labels, self.__feature_names = train_feats, train_labels, feat_names
        self.__train_data = [np.append(feats,[label]) for feats, label in zip(train_feats, train_labels)]
        self.__tree = self.__build_tree(self.__train_data, 0)

    # Print tree
    def print_tree(self):
        if self.__tree is None:
            raise Exception("Training Data not fitted.")

        print("Decision Tree: \n" + str(self.__tree) + "\n")

    # Predict given input data
    def predict(self, test_feats, test_labels):
        if self.__train_features is None or self.__train_labels is None:
            raise Exception("Training Data not fitted.")

        self.__test_features, self.__test_labels = test_feats, test_labels
        self.__predictions = [self.__tree.classify(row) for row in test_feats]
        return self.__predictions

    # Score model
    def score(self):
        if self.__train_features is None or self.__train_labels is None:
            raise Exception("Training Data not fitted.")
        if self.__test_features is None or self.__test_labels is None:
            raise Exception("Test data not inputted.")

        count = 0
        for prediction, result in zip(self.__predictions, self.__test_labels):
            if prediction == result:
                count += 1
        confusion_matrix = pd.crosstab(pd.Series(self.__predictions, name="Predicted"),
                                       pd.Series(self.__test_labels, name="Actual"))
        print("Accuracy: {0:.2f}%".format(count / len(self.__test_labels) * 100))
        print(confusion_matrix, "\n")
