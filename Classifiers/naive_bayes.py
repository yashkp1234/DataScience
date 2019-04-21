import numpy as np
import pandas as pd
from math import exp, sqrt, pi
import operator


def gaussian_probability(x, mean, stdev):
    if stdev == 0: stdev = 0.01
    exponent = exp(-1 * (x - mean) ** 2 / (2 * stdev ** 2))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


class NaiveBayes:
    # Constructor
    def __init__(self, distribution="gaussian"):
        self.__test_features = None
        self.__test_labels = None
        self.__train_features = None
        self.__train_labels = None
        self.__predictions = None
        self.__distribution = distribution
        self.__class_sep_data = None
        self.__class_prob = None
        print("Naive Bayes Initiated Using using " + distribution + " distribution.\n")

    # Calculate probability using inputted distribution
    def probability_function(self, x, mean, std):
        # For now no other distributions implemented
        if self.__distribution:
            return gaussian_probability(x, mean, std)

    # Separate each class into a dataset
    def separate_classes(self):
        unique_labels = set(self.__train_labels)
        self.__class_sep_data = dict((class_name, []) for class_name in list(unique_labels))
        self.__class_prob = dict((class_name, 1) for class_name in list(unique_labels))
        for row in zip(self.__train_features, self.__train_labels):
            self.__class_sep_data[row[1]].append(row[0])

    # Calculate mean and standard deviation for each attribute in each class
    def summarize_data(self):
        for key in self.__class_sep_data:
            summary_val = []
            for attr_vals in zip(*self.__class_sep_data[key]):
                np_attr_val = np.array(attr_vals)
                summary_val.append((np_attr_val.mean(), np_attr_val.std()))
            self.__class_sep_data[key] = summary_val

    # Calculate probability of test sample being in each class and output class with largest prob. as prediction
    def calculate_probabilities_and_predict(self):
        for row in self.__test_features:
            prob_dict = self.__class_prob.copy()
            for key in self.__class_sep_data:
                probabilities = self.__class_sep_data[key]
                for val, prob in zip(row, probabilities):
                    prob_dict[key] *= self.probability_function(val, prob[0], prob[1])
            self.__predictions.append(max(prob_dict.items(), key=operator.itemgetter(1))[0])

    # Fit the training data on to the model
    def fit(self, train_features, train_labels):
        self.__train_features, self.__train_labels = list(train_features), list(train_labels)
        self.separate_classes()
        self.summarize_data()

    # Run the prediction set-up and functions
    def predict(self, test_features, test_labels):
        if not self.__train_features or not self.__train_labels:
            raise Exception("Training Data not fitted.")

        self.__predictions = []
        self.__test_features, self.__test_labels = list(test_features), list(test_labels)
        self.calculate_probabilities_and_predict()

    # Output score which includes accuracy and confusion matrix
    def score(self):
        if not self.__train_features or not self.__train_labels:
            raise Exception("Training data not fitted.")
        if not self.__test_labels or not self.__test_features:
            raise Exception("Test data not inputted.")

        count = 0
        for prediction, result in zip(self.__predictions, self.__test_labels):
            if prediction == result:
                count += 1
        confusion_matrix = pd.crosstab(pd.Series(self.__predictions, name="Predicted"),
                                       pd.Series(self.__test_labels, name="Actual"))
        print("Accuracy: {0:.2f}%".format(count / len(self.__test_labels) * 100))
        print(confusion_matrix, "\n")
