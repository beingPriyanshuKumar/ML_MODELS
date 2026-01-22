# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#traing the simple regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the training set results
y_pred=regressor.predict(X_test)
plt.scatter(X_train,y_train , color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue' )
plt.xlabel('years of experince')
plt.ylabel('salary')
plt.show()

#predicting the test set results
y_pred=regressor.predict(X_test)
plt.scatter(X_test,y_test , color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue' )
plt.xlabel('years of experince')
plt.ylabel('salary')
plt.show()