from sklearn.model_selection import train_test_split
from Classifiers.random_forest import RandomForest
import pandas as pd
import matplotlib.pyplot as plt

def my_round(x, base=10):
    return base * round(x/base)

income_eval = pd.read_csv("../Datasets/income_evaluation.csv")
print(income_eval.describe().to_string())
print(income_eval.head().to_string())

income_eval["age"] = income_eval["age"].apply(lambda x: my_round(x))
#income_eval[" fnlwgt"] = income_eval[" fnlwgt"].apply(lambda x: my_round(x, 20000))
income_eval[" capital-gain"] = income_eval[" capital-gain"].apply(lambda x: my_round(x, 500))
income_eval[" capital-loss"] = income_eval[" capital-loss"].apply(lambda x: my_round(x, 500))
income_eval[" hours-per-week"] = income_eval[" hours-per-week"].apply(lambda x: my_round(x, 10))

income_eval = income_eval.drop(" education-num", axis=1)

target = income_eval[" income"].values
features = income_eval.drop([" income", " workclass", " marital-status",
                             " race", " sex", " native-country", " fnlwgt"], axis=1)
column_names = list(features.columns)

X_train, X_test, y_train, y_test = train_test_split(features.values, target,
                                                    test_size=0.33, random_state=25, shuffle=True)

rf = RandomForest(max_depth=5, max_features='sqrt', n_estimators=30, min_sample_leaf=0.1)
rf.fit(X_train, y_train, column_names)
rf.predict(X_test)
rf.score(y_test)
df = rf.feature_importance()
print(df)
df.plot.barh(x='Features', y='Importance')
plt.show()


"""
                age        fnlwgt   education-num   capital-gain   capital-loss   hours-per-week
count  32561.000000  3.256100e+04    32561.000000   32561.000000   32561.000000     32561.000000
mean      38.581647  1.897784e+05       10.080679    1077.648844      87.303830        40.437456
std       13.640433  1.055500e+05        2.572720    7385.292085     402.960219        12.347429
min       17.000000  1.228500e+04        1.000000       0.000000       0.000000         1.000000
25%       28.000000  1.178270e+05        9.000000       0.000000       0.000000        40.000000
50%       37.000000  1.783560e+05       10.000000       0.000000       0.000000        40.000000
75%       48.000000  2.370510e+05       12.000000       0.000000       0.000000        45.000000
max       90.000000  1.484705e+06       16.000000   99999.000000    4356.000000        99.000000
   age          workclass   fnlwgt   education   education-num       marital-status          occupation    relationship    race      sex   capital-gain   capital-loss   hours-per-week  native-country  income
0   39          State-gov    77516   Bachelors              13        Never-married        Adm-clerical   Not-in-family   White     Male           2174              0               40   United-States   <=50K
1   50   Self-emp-not-inc    83311   Bachelors              13   Married-civ-spouse     Exec-managerial         Husband   White     Male              0              0               13   United-States   <=50K
2   38            Private   215646     HS-grad               9             Divorced   Handlers-cleaners   Not-in-family   White     Male              0              0               40   United-States   <=50K
3   53            Private   234721        11th               7   Married-civ-spouse   Handlers-cleaners         Husband   Black     Male              0              0               40   United-States   <=50K
4   28            Private   338409   Bachelors              13   Married-civ-spouse      Prof-specialty            Wife   Black   Female              0              0               40            Cuba   <=50K
Random Forest initiated.

Accuracy: 82.13%
Actual      <=50K   >50K
Predicted               
 <=50K       8193   1882
 >50K          38    633 

Features
 hours-per-week     3.962667
age                 4.469333
 capital-loss       8.235333
 education          8.664000
 occupation        11.431333
 relationship      20.424000
 capital-gain      42.813333
Name: Importance, dtype: float64
"""


