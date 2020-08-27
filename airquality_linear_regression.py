# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:11:04 2020

@author: khkre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("D:\\AirQualityUCI (1).csv",sep=';')
df.info()
df.drop(df.columns[[-1,-2]],axis=1,inplace=True)
df.isnull().any()
df.isnull().sum()
df.dropna(axis=0,inplace=True)
df.replace(',','.',regex=True,inplace=True)
df.describe()
df.drop(df.columns[[0,1]],axis=1,inplace=True)
df['CO(GT)']=df['CO(GT)'].astype('float64')
df['C6H6(GT)']=df['C6H6(GT)'].astype('float64')
df['T']=df['T'].astype('float64')
df['RH']=df['RH'].astype('float64')
df['AH']=df['AH'].astype('float64')
df.replace(-200,np.nan,regex=True,inplace=True)
df.fillna(df.mean(),inplace=True)

x=df.drop(df.columns[-2],axis=1)
y=df[['RH']]

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x=sc_x.fit_transform(x)

sc_y = StandardScaler()
y=sc_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

ypred=lr.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn import metrics
print('MSE:',metrics.mean_squared_error(y_test,ypred))












