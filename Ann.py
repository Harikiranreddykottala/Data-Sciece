import pandas as pd
import numpy as np
df=pd.read_csv("D:\Churn_Modelling.csv")
x=df.iloc[:,3:13]
y=df.iloc[:,-1]

geography=pd.get_dummies(x['Geography'],drop_first=True)
gender=pd.get_dummies(x['Gender'],drop_first=True)

x=pd.concat([x,geography,gender],axis=1)
x=x.drop(['Geography','Gender'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

import keras
from keras.models import Sequential
#from keras.layers import ReLU,LeakyRelu
from keras.layers import Dense
from keras .layers import Dropout

classifier=Sequential()
'''#classifier.add(Dense(output_dim=6,init='he_uniform',activation='relu',input_dim=11))
#Input layer and 1st hidden layer,units=output_dim,
kernel_initializer=optimizser ref='''
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))

#2nd Hidden Layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))

#output layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

classifier.summary()
'''optimizer adam is best and for loss,if there are are only two output values like 
0 and 1 use binary_crossentropy if there are multiple output categories then use
categorical_crossentropy'''
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')

model_history=classifier.fit(x_train,y_train,batch_size=10,epochs=100,validation_split=0.33)

y_pred=classifier.predict(x_test)
y_pred=y_pred>0.5

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)

