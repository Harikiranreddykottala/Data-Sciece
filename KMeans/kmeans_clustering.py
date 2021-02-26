# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 07:02:15 2020

@author: khkre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("E:\\Tharun\\Data-science---Machine-Learning-projects-main\\model_selections-master\\Purchased_Dataset.csv")
df.head()
x=df.iloc[:,1:]
x.isnull().sum()
y=pd.get_dummies(x['Gender'],drop_first=True)
x['Gender']=y
"""Hopkins test decides the tendency of cluster whether it will form into clusters or not"""
#Calculating the Hopkins statistic
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
 
def hopkins(x):
    d = x.shape[1]
    n = len(x) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(x.values) 
    rand_X = sample(range(0, n, 1), m) 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(x,axis=0),np.amax(x,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(x.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1]) 
    HO = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(HO):
        print(ujd, wjd)
        HO = 0
 
    return HO

for i in range(10):

    print(hopkins(x))
    
sns.boxplot(x['EstimatedSalary'])
sns.boxplot(x['Age'])

"""within sum of clusters is used to decide the k value """

from sklearn.cluster import KMeans
wcss=[]  # wcss is a empty list indicates within cluster sum of squares
k=range(1,15)
for i in k:
    kmeans=KMeans(i,random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(k,wcss)                      #plot the diagram in the form of x and y 
plt.title("Find optimal k value")
plt.xlabel("k value")
plt.ylabel("wcss value : within cluster sum of squares")
plt.show()


kmeans=KMeans(4,random_state=42)
kmeans.fit(x)
y_kmeans=kmeans.fit_predict(x)
y_kmeans

x['cluster']=y_kmeans
x.head()

x.cluster.value_counts()

from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
""" silhouette score helps in understanding which k value fits better to the data"""
from sklearn.metrics import silhouette_score
range_clusters=[2,3,4,5,6,7,8]
for i in range_clusters:
    kmeans=KMeans(i,random_state=42)
    kmeans.fit(x)
    cluster_labels=kmeans.labels_
    silhouette=silhouette_score(x,cluster_labels)
    print(silhouette)

x.head()
sns.scatterplot(x = 'Age', y ='EstimatedSalary', hue = 'cluster', data =x,palette=['blue','green','red','black'])
plt.xlabel('Age',fontsize=13)
plt.ylabel('EstimatedSalary',fontsize=13)
plt.legend()
plt.show()

sns.scatterplot(x = 'Gender', y ='EstimatedSalary', hue = 'cluster', data =x,palette=['blue','green','red','black'])
plt.xlabel('Gender',fontsize=13)
plt.ylabel('EstimatedSalary',fontsize=13)
plt.legend()
plt.show()

'''salary above 1L from both gender fall in cluster 3,70k-100k in cluster-2,40k-60k in cluster-0
salary 0-40k in cluster1'''


x.groupby(['Gender','EstimatedSalary'])['cluster'].value_counts()

""" Its pretty clear salary has played major role here in forming different clusters
Gender and age has very less effect in this dataset"""


