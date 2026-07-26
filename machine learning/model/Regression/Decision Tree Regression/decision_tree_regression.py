# Decision Tree Regression

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# Training 
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (higher resolution)
