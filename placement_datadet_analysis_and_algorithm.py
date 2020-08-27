# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 07:14:36 2020

@author: khkre
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
placement=pd.read_csv("D:\\datasets_596958_1073629_Placement_Data_Full_Class.csv")
placement['status']=placement['status'].replace('Placed',1)
placement['status']=placement['status'].replace('Not Placed',0)
placement.info()
placement.describe()
placement['status'].value_counts()
(placement['status']==1).isnull().sum()
(placement['salary']).isnull().sum()
placement['salary'].replace(np.nan,0,inplace=True)
placement.salary.describe()
sns.countplot(placement['gender'],hue=placement['status'])
placement.groupby(['gender','status']).count()['salary']
''' Total Female=76
 63% got placed-48 members
 
 total male=139
 71% got placed-100 mem
 
thus we can say there is no significant amount of biasing in gender'''

placement.groupby(['ssc_p','status']).count()['salary']
sns.swarmplot(x='status',y='ssc_p',hue='gender',data=placement)
''' 50% above min to get placement in ssc for any gender'''
sns.swarmplot(x='status',y='hsc_p',hue='gender',data=placement)
''' 50% above min to get placement in hsc for any gender'''
sns.swarmplot(x='status',y='degree_p',hue='gender',data=placement)
''' above 55% is ok but still its not aguarantee factor'''
sns.swarmplot(x='status',y='etest_p',hue='gender',data=placement)
''' percentage doesn't matter'''
sns.swarmplot(x='status',y='mba_p',hue='gender',data=placement)
''' score doesnt matter'''
sns.barplot(x='status',y='hsc_p',hue='hsc_b',data=placement)
'''hsc board doesnt matter'''
sns.barplot(x='status',y='hsc_p',hue='hsc_s',data=placement)
'''hsc_s stream doesn't matter'''
sns.barplot(x='status',y='degree_p',hue='degree_t',data=placement)
''' degree_t stream doesnt matter'''
sns.barplot(x='status',y='workex',data=placement)
'''work exp matters in placement'''
sns.barplot(y='status',x='specialisation',data=placement)
''' mkt and fin is better for ooportunities'''

#linear regression
'''x=placement.drop(['sl_no','gender','ssc_b','hsc_b','hsc_s','degree_t',],axis=1)
cat_features=['workex']
x=pd.get_dummies(x,columns=cat_features,drop_first=True)
x.drop(['salary'],axis=1,inplace=True)
y=placement['salary']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)

coef=lm.coef_
intercept=lm.intercept_

y_pred=lm.predict(x_test)
from sklearn import metrics

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))'''

x=placement.drop(['sl_no','gender','ssc_b','hsc_b','hsc_s','degree_t','status','salary'],axis=1)
cat_features=['workex']
x=pd.get_dummies(x,columns=cat_features,drop_first=True)
cat_features1=['specialisation']
x=pd.get_dummies(x,columns=cat_features1,drop_first=True)

y=placement['status']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)



from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)
predictions=dtree.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))



from sklearn.ensemble import RandomForestClassifier
rtc=RandomForestClassifier(n_estimators=500)
rtc.fit(x_train,y_train)

predictions=rtc.predict(x_test)
print('confusion_matrix:',confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))



























