# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
os.getcwd()

# Importing the dataset
dataset = pd.read_csv('C:\\MLAZ\\Part 1 - Data Preprocessing\\Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation  import train_test_split
x_train,y_train, x_test, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_x.transform(y_train)
"""