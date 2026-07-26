import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Market_Basket_Optimisation.csv')
transaction = []
for i in range (1,7051):
    for j in range (1,20):
        transaction.append(str(dataset.values[i,j]))


