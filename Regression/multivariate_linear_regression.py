import numpy as np
from sklearn.metrics import r2_score


class MultiVariateLinearRegression:
    # Constructor, batch parameter allows to switch between stochastic and batch gradient descent
    def __init__(self, learning_rate=0.01, epoch=None, epsilon=None, batch=False, print_mod=10):
        if not epoch and not epsilon:
            raise Exception("You must either have an epoch limit or a limit on the cost function.")

        if not epoch:
            epoch = 1000

        self.__test_features = None
        self.__test_labels = None
        self.__train_features = None
        self.__train_labels = None
        self.__predictions = None
        self.__coefficients = None
        self.__print_mod = print_mod
        self.__num_epoch = epoch
        self.__learning_rate = learning_rate
        self.__epsilon = epsilon
        self.__cost_list = []
        self.__batch = batch
        print("Multivariate Regression Initiated.\n")

    # Get the cost list
    def get_cost_list(self):
        return self.__cost_list

    # Get coefficients for each feature
    def get_coefficients(self):
        return self.__coefficients

    # Calculate the predictions
    def __calculate_predictions(self):
        return self.__test_features @ np.array(self.__coefficients).T

    # Calculate the cost function
    def __cost_function(self):
        cost = sum(((self.__train_features @ np.array(self.__coefficients).T) - self.__train_labels) ** 2) \
               / (2 * len(self.__train_features))
        self.__cost_list.append(cost)
        return cost

    # Update the coefficients for each feature
    def __update_coefficients(self, num=1):
        new_coefficients = []
        for coef, index in zip(self.__coefficients, range(0, len(self.__coefficients))):
            summation = 0
            training_data = zip(self.__train_features, self.__train_labels)

            if self.__batch:
                for row_feat, row_targ in training_data:
                    summation += (self.__coefficients.dot(row_feat) - row_targ) * row_feat[index]
                new_coefficients.append(coef - (summation * self.__learning_rate / len(self.__train_features)))

            else:
                row_feat, row_targ = self.__train_features[num], self.__train_labels[num]
                summation += (self.__coefficients.dot(row_feat) - row_targ) * row_feat[index]
                new_coefficients.append(coef - (summation * self.__learning_rate / len(self.__train_features)))

        self.__coefficients = np.array(new_coefficients)

    # Run the Gradient Descent Algorithm
    def _grad_descent_calc(self):
        self.__coefficients = np.array([1 for _ in range(len(self.__train_features[0]))])
        cost = None
        for loop_num in range(self.__num_epoch):

            if cost and self.__epsilon and cost <= self.__epsilon:
                break

            self.__update_coefficients(loop_num % len(self.__train_features))
            cost = self.__cost_function()
            if loop_num % self.__print_mod == 0:
                print("Iteration: {}, Cost Function: {}".format(loop_num, cost))

        if self.__epsilon and not self.__num_epoch and cost > self.__epsilon:
            print("Stopped due to max limit on epochs.")

    # Fit training data to model
    def fit(self, x, y):
        x = [np.append([1], val) for val in x]
        self.__train_features, self.__train_labels = np.array(x), np.array(y)
        self._grad_descent_calc()

    # Predict results for input test data
    def predict(self, test_x, test_y):
        if self.__train_features is None or self.__train_labels is None:
            raise Exception("Training Data not fitted.")

        test_x = [np.append([1], val) for val in test_x]
        self.__test_features, self.__test_labels = test_x, test_y
        self.__predictions = self.__calculate_predictions()
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
