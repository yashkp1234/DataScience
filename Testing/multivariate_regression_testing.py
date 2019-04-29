import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Regression.multivariate_linear_regression import MultiVariateLinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
df = pd.DataFrame(boston.data)

features = preprocessing.scale(df.values)
target = boston.target
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=10, shuffle=True)

mvr = MultiVariateLinearRegression(learning_rate=0.3, epoch=5000, epsilon=2, print_mod=500, batch=False)
mvr.fit(X_train, y_train)
predictions = mvr.predict(X_test, y_test)
mvr.score()
plt.scatter(list(range(0, len(mvr.get_cost_list()))), mvr.get_cost_list())
plt.ylabel('Cost')
plt.xlabel('Epoch Number')
plt.title('Cost Function Over Iterations')
plt.show()
plt.scatter(predictions, y_test)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


"""
Results Below, other than plots:

Multivariate Regression Initiated.

Iteration: 0, Cost Function: 286.9905904059428
Iteration: 500, Cost Function: 112.28886033018048
Iteration: 1000, Cost Function: 54.20842566051757
Iteration: 1500, Cost Function: 29.57581598553555
Iteration: 2000, Cost Function: 19.33559666043571
Iteration: 2500, Cost Function: 14.733699289626747
Iteration: 3000, Cost Function: 12.750100383558548
Iteration: 3500, Cost Function: 11.73307480016693
Iteration: 4000, Cost Function: 11.33006094794954
Iteration: 4500, Cost Function: 11.012803149929384
R Squared Score: 0.70
"""