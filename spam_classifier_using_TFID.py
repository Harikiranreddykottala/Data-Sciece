# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:22:12 2020

@author: khkre
"""

import pandas as pd
messages=pd.read_csv("D:\\spam dataset\\SMSSpamCollection",sep='\t',names=['labels','message'])

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps=PorterStemmer()
wordnet=WordNetLemmatizer()
corpus=[]

for i in range(0,len(messages['message'])):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
x=tf.fit_transform(corpus).toarray()
y=messages['labels']
y=pd.get_dummies(y,drop_first=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)

y_pred=spam_detect_model.predict(x_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

