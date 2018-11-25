# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 04:18:08 2018

@author: Vader
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


""" 
No need to split - Only 10 records
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = y.reshape((len(y), 1))
y = sc_y.fit_transform(y)

## Fitting SVR model
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)
## Predicting a new result with SVR
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

##--- Visualizing SVR results
plt.scatter(X,y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel("Position level")
plt.ylabel('Salary')
plt.show()

