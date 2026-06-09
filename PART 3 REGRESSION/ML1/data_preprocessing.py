import numpy as np
import matplotlib.pyplot as plt       
import pandas as pd                      # importing importanta datas


dataset = pd.read_csv('Data.csv')
x= dataset.iloc[:, :-1].values     #separting daabase acc to categorial data
y= dataset.iloc[:,-1].values
print(x)
print(y)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')       #filling up the misssing values
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])


from sklearn.compose import ColumnTransformer  #encoding categorial data
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)  #encoding the dependent variable
print(y)


from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train= train_test_split(x,y,test_size=0.2, random_state=1) #seprating the training and testing data


from sklearn.preprocessing import StandardScaler # feature scaling using standardistation
sc = StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
x_test[:,3:]=sc.transform(x_test[:,3:])  
print(x_train)
print(x_test)