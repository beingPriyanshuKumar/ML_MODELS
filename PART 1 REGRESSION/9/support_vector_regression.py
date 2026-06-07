# Support Vector Regression (SVR)

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
print (x)
print (x)
y = y.reshape(len(y),1)
print (y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
sc_y = StandardScaler()
X_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train,y_train)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[2]])).reshape(-1,1))
np.set_printoptions(precision=2)

# # Visualising the SVR results
# plt.scatter(sc_x.inverse_transform(x_train), sc_y.inverse_transform(y), color = 'red')
# plt.plot(sc_x.inverse_transform(x_train), sc_y.inverse_transform(regressor.predict(sc_x.transform(x_train))), color = 'blue')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)