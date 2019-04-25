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
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=1)

mvlr = MultiVariateLinearRegression(learning_rate=0.3, epoch=5000, epsilon=2, print_mod=100, batch=False)
mvlr.fit(X_train, y_train)
predictions = mvlr.predict(X_test, y_test)
mvlr.score()
plt.scatter(list(range(0, len(mvlr.get_cost_list()))), mvlr.get_cost_list())
plt.ylabel('Cost')
plt.xlabel('Epoch Number')
plt.title('Cost Function Over Iterations')
plt.show()
plt.scatter(predictions, y_test)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

"""
Multivariate Regression Initiated.

Iteration: 0, Cost Function: 293.4497088162596
Iteration: 100, Cost Function: 226.63070070944192
Iteration: 200, Cost Function: 188.82911014818632
Iteration: 300, Cost Function: 155.77118192217844
Iteration: 400, Cost Function: 133.59062247459062
Iteration: 500, Cost Function: 114.67489216625222
Iteration: 600, Cost Function: 97.5619797046863
Iteration: 700, Cost Function: 84.44639166519235
Iteration: 800, Cost Function: 72.59949020395082
Iteration: 900, Cost Function: 64.40767514458175
Iteration: 1000, Cost Function: 55.54871250024841
Iteration: 1100, Cost Function: 48.70441251917344
Iteration: 1200, Cost Function: 43.561571851901085
Iteration: 1300, Cost Function: 38.05211024854222
Iteration: 1400, Cost Function: 34.309146552680374
Iteration: 1500, Cost Function: 31.267382465506522
Iteration: 1600, Cost Function: 28.44017804133289
Iteration: 1700, Cost Function: 26.008970051150015
Iteration: 1800, Cost Function: 23.68642415405029
Iteration: 1900, Cost Function: 22.30567510186054
Iteration: 2000, Cost Function: 20.60771784075105
Iteration: 2100, Cost Function: 19.534792793410762
Iteration: 2200, Cost Function: 18.63161625393147
Iteration: 2300, Cost Function: 17.50426599809769
Iteration: 2400, Cost Function: 16.902738663197013
Iteration: 2500, Cost Function: 16.241447956157796
Iteration: 2600, Cost Function: 15.752683700519357
Iteration: 2700, Cost Function: 15.16837398448545
Iteration: 2800, Cost Function: 14.814914226401394
Iteration: 2900, Cost Function: 14.520511697936291
Iteration: 3000, Cost Function: 14.067827101436642
Iteration: 3100, Cost Function: 13.875542288559856
Iteration: 3200, Cost Function: 13.755151037717296
Iteration: 3300, Cost Function: 13.542665536875402
Iteration: 3400, Cost Function: 13.340255962313352
Iteration: 3500, Cost Function: 13.178314107082974
Iteration: 3600, Cost Function: 13.107622770400187
Iteration: 3700, Cost Function: 12.943489311510138
Iteration: 3800, Cost Function: 12.856864089237686
Iteration: 3900, Cost Function: 12.825573503919085
Iteration: 4000, Cost Function: 12.682894975610534
Iteration: 4100, Cost Function: 12.605799928744943
Iteration: 4200, Cost Function: 12.594856180299095
Iteration: 4300, Cost Function: 12.533697772204878
Iteration: 4400, Cost Function: 12.452676119799587
Iteration: 4500, Cost Function: 12.442353321941692
Iteration: 4600, Cost Function: 12.42390813646088
Iteration: 4700, Cost Function: 12.325089137448535
Iteration: 4800, Cost Function: 12.321134296368776
Iteration: 4900, Cost Function: 12.325788609511703
R Squared Score: 0.75
"""