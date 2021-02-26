# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:29:03 2020

@author: denny
"""

'條件: 數值型且需標準化，無NA'
from sklearn import cluster, datasets, metrics, preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%%
'分幾群'
data = pd.read_csv('D:/train_new.csv', encoding='utf-8')
data = pd.read_csv('D:/test_new.csv', encoding='utf-8')
#numeric = data[["bone_length", "has_soul", "hair_length", "rotting_flesh"]]
#iris = datasets.load_iris()
#numeric = iris.data
#%%
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(data) #numeric
silhouette_avgs = []
ks = range(2, 11) 
for k in ks:
    kmeans_fit = cluster.KMeans(n_clusters = k).fit(X)
    cluster_labels = kmeans_fit.labels_
    silhouette_avg = metrics.silhouette_score(X, cluster_labels)
    silhouette_avgs.append(silhouette_avg)

#近1好，近-1差
plt.bar(ks, silhouette_avgs)
plt.show()
print(silhouette_avgs)
#%%
'選距離&中心'
km = cluster.KMeans(n_clusters=6,
            init= 'k-means++', #中心選法，random
            n_init=10, #隨機心做10次k-means，取sse最小的
            max_iter=300, #最大迭代次數
            tol=1e-04, #可容許誤差
            random_state=0)
y_km = km.fit_predict(X) #done
#%%
'視覺化'
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s' , edgecolor='black',
            label='cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker= 'o', edgecolor='black',
            label='cluster 2')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red' , edgecolor='black',
            label=' cent roids ')
plt. legend(scatterpoints=1)
plt.grid()
plt. show()
#%%
'合併&輸出'
y_km = pd.DataFrame(data=y_km,columns=["group"])
final = pd.concat([data, y_km], axis=1)
#final.to_csv('D:/train_final.csv',encoding='utf_8',index=False)
final.to_csv('D:/test_final.csv',encoding='utf_8',index=False)
