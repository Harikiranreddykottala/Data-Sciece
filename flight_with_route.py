# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel("C:\\Users\\khkre\\Downloads\\Data_Train.xlsx")
df.info()
df.describe()
df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'])
sns.distplot(df['Price'])
plt.figure(figsize=(8,10))
sns.barplot(x='Date_of_Journey',y='Price',data=df)
df['Date_of_Journey']=df['Date_of_Journey'].view('int64')


df['Dep_Time']=pd.to_datetime(df['Dep_Time'])
df['Dep_Time']=df['Dep_Time'].apply(lambda x:x.replace(year=2019))
df['Dep_Time']=df['Dep_Time'].view('int64')

df['Arrival_Time']=pd.to_datetime(df['Arrival_Time'])
df['Arrival_Time']=df['Arrival_Time'].apply(lambda x:x.replace(year=2019))
df['Arrival_Time']=df['Arrival_Time'].view('int64')



new=df['Duration'].str.split(' ',n=2,expand=True)
df['Hour']=new[0]
df['Minutes']=new[1]
df.drop(columns=['Duration'],axis=1,inplace=True)


df['Hour'].replace('h',' ',regex=True,inplace=True)
df[df['Hour']=='5m']
df.drop(index=6474,axis=0,inplace=True)
df['Hour']=df['Hour'].astype('float64')
df['Hour']=60*df['Hour']

df['Minutes'].replace('None',0,inplace=True,regex=True)
df['Minutes']=df['Minutes'].fillna(0)
df['Minutes'].replace('m',' ',inplace=True,regex=True)
df['Minutes']=df['Minutes'].astype('float64')

df['duration_in_minutes']=df['Hour']+df['Minutes']
df.drop_duplicates(inplace=True)
df.isnull().any()
df.isnull().sum()
df=df.dropna(axis=0)
df.Airline.unique()
plt.figure(figsize=(12,10))
sns.barplot(x='Airline',y='Price',data=df)
df.Airline.nunique()#12
df.Source.nunique()#5
df.Destination.nunique()#6
df.Total_Stops.nunique()#5
df.columns
'''['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route',
       'Dep_Time', 'Arrival_Time', 'Total_Stops', 'Additional_Info', 'Price',
       'Hour', 'Minutes', 'duration_in_minutes']'''
x=df[['Airline','Source','Destination','Route','Total_Stops','Date_of_Journey','Dep_Time', 'Arrival_Time','duration_in_minutes']]
cat_features=['Airline','Source', 'Destination','Total_Stops','Route']
x['Date_of_Journey']=x['Date_of_Journey'].view('int64')
x=pd.get_dummies(x,columns=cat_features,drop_first=True)
y=df[['Price']]

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

y_pred=lr.predict(x_test)

from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score
print('R2_score:',r2_score(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

from sklearn.ensemble import RandomForestRegressor
model_1 = RandomForestRegressor(n_estimators=100, random_state=1)
model_1.fit(x_train, y_train)
preds = model_1.predict(x_test)
print('R2_score:',r2_score(y_test,preds))
print('MSE:',metrics.mean_squared_error(y_test,preds))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,preds)))

from sklearn.ensemble import RandomForestRegressor
model_2 = RandomForestRegressor(n_estimators=200, random_state=1)
model_2.fit(x_train, y_train)
preds1 = model_2.predict(x_test)
print('R2_score:',r2_score(y_test,preds1))
print('MSE:',metrics.mean_squared_error(y_test,preds1))
