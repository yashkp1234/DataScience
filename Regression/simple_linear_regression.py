import numpy as np
from sklearn.metrics import r2_score


# Create linear regression function based on inputs
def make_regression_function(cov, var, x_mean, y_mean):
    def regression(val):
        return cov / var * val + y_mean - x_mean * (cov / var)
    return regression


class SimpleLinearRegression:
    # Constructor
    def __init__(self):
        self.__test_features = None
        self.__test_labels = None
        self.__train_features = None
        self.__train_labels = None
        self.__predictions = None
        self.__regression_func = None
        print("Linear Regression Initiated.\n")

    # Create the regression function from training dataset
    def calculate_regression(self):
        cov = np.cov(self.__train_features, self.__train_labels)[1][0]
        var_x = np.var(self.__train_features)
        x_mean, y_mean = np.mean(self.__train_features), np.mean(self.__train_labels)
        self.__regression_func = make_regression_function(cov, var_x, x_mean, y_mean)

    # Fit training data to model
    def fit(self, x, y):
        self.__train_features, self.__train_labels = np.array(x), np.array(y)
        self.calculate_regression()

    # Predict results for input test data
    def predict(self, test_x, test_y):
        if self.__train_features is None or self.__train_labels is None:
            raise Exception("Training Data not fitted.")

        self.__test_features, self.__test_labels = test_x, test_y
        self.__predictions = [self.__regression_func(val) for val in test_x]
        return self.__predictions

    # Score model using Root Mean Square Error
    def score(self):
        if self.__train_features is None or self.__train_labels is None:
            raise Exception("Training Data not fitted.")
        if self.__test_features is None or self.__test_labels is None:
            raise Exception("Test data not inputted.")

        score = r2_score(self.__test_labels, self.__predictions)
        print("R Squared Score: {0:.2f}".format(r2_score(self.__test_labels, self.__predictions)))
        return score

