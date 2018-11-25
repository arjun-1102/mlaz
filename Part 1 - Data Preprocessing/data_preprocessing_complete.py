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

"""
#Dealing with missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(x[:,1:3])
x[:, 1:3]=imputer.transform(x[:, 1:3])

#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()


labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
"""

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