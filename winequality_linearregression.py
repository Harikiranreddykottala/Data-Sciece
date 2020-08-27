# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
wine=pd.read_csv("C:\\Users\\khkre\\Downloads\\winequality.csv")
wine.info()
wine.describe()
wine.isnull().any()
wine.fillna(method='ffill')
wine.columns
plt.figure(figsize=(12,7))
x=wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
sns.pairplot(x)
''' kindly check all the graphs and take necessary fetures anly to get best results'''

y=wine['quality']
sns.distplot(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)

coef=lm.coef_
intercept=lm.intercept_

y_pred=lm.predict(x_test)

from sklearn import metrics

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
''' 0.63 is slightly greater than 10% of mean 5.63 of overall quality but still
our algorithm can give reasonbly better results.kindly note RMSE should be less than 
10% of overall target variable mean'''