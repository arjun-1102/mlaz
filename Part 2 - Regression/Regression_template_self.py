# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 13:12:15 2018

@author: Vader
"""

### Python regreeion template
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 00:31:08 2018

@author: Vader
"""

### Polynomial regression
# Data Preprocessing Template

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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

## Fitting non-linear regression model


## Predicting a new result with non-linear regression
y_pred = regressor.predict(6.5)

##--- Visualizing regression results
plt.scatter(X,y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel("Position level")
plt.ylabel('Salary')
plt.show()

##--- Visualizing regression results - high res
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.resharpe((len(X_grid), 1))
plt.scatter(X,y, color='red')
plt.plot(X, regressor.predict(X_grid), color='blue')
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel("Position level")
plt.ylabel('Salary')
plt.show()

