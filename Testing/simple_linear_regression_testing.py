import pandas as pd
from sklearn.model_selection import train_test_split
from Regression.simple_linear_regression import SimpleLinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv(r"..\Datasets\Auto_Insurance.csv")
SLR = SimpleLinearRegression()


features = df["X"].values  # Number of claims
target = df["Y"].values  # Total payment

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.40, random_state=12)

SLR.fit(X_train, y_train)
predictions = SLR.predict(X_test, y_test)
SLR.score()
print(predictions[:10])
print(y_test[:10])
print(r2_score(y_test, predictions))


"""
Results Below:

Linear Regression Initiated.

R Squared Score: 0.62
 actual:   
   [34.41, 52.22, 41.53, 180.40, 45.09, 
    70.02, 216.01, 59.34, 30.85, 126.99]
    
predicted:
    [38.1  87.4  50.9 214. 27.9  
     77.5 162.8  21.3  13.2 194.5]

"""