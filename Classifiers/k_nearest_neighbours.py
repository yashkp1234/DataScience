from math import sqrt
import operator
import pandas as pd


def euclidean_distance(x, y):
    return sqrt(sum((i - k) ** 2 for i,k in zip(x,y)))


def manhattan_distance(x, y):
    return sum(abs(i - k) for i,k in zip(x,y))


def get_prediction(neighbours):
    votes = {}
    for row in neighbours:
        key = row[1]
        if key in votes:
            votes[key] += 1
        else:
            votes[key] = 1
    return max(votes.items(), key=operator.itemgetter(1))[0]


class KNearestNeighbours:
    # Constructor
    def __init__(self, k=1, calc_method="euclidean"):
        self.__k = k
        self.__test_features = None
        self.__test_labels = None
        self.__train_features = None
        self.__train_labels = None
        self.__predictions = None
        self.__scores = None
        self.__calc_method = calc_method
        print("KNN Initiated Using k=" + str(k) + " using " + calc_method + " distance.\n")

    # Calculate distance between points based on inputted method
    def __distance(self, x, y):
        if self.__calc_method == "manhattan":
            return manhattan_distance(x, y)
        else:
            return euclidean_distance(x, y)

    # Retrieve the K nearest neighbours
    def __k_nearest_neighbours(self, test_sample):
        scores = [[self.__distance(row[1], test_sample), row[0]]
                  for row in zip(self.__train_labels, self.__train_features)]
        scores.sort(key=lambda x: x[0])
        self.__scores = scores
        return self.__scores[:self.__k]

    # Fit the training data on to the model
    def fit(self, train_features, train_labels):
        self.__train_features, self.__train_labels = list(train_features), list(train_labels)

    # Predict results for inputted test samples
    def predict(self, test_features, test_labels):
        if self.__train_features is None or self.__train_labels is None:
            raise Exception("Training Data not fitted.")

        self.__test_features, self.__test_labels = list(test_features), list(test_labels)
        neighbours_list = [self.__k_nearest_neighbours(row) for row in self.__test_features]
        self.__predictions = [get_prediction(neighbours) for neighbours in neighbours_list]
        return self.__predictions

    # Output score which includes accuracy and confusion matrix
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



