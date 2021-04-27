import math

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


filename = 'car.csv'

df = pd.read_csv(filename, na_values="?" )
print(df.head())
print(df.shape)

#df['buying'] = df['buying'].astype('category')
#df['maint'] = df['maint'].astype('category')
#df['doors'] = df['doors'].astype('category')
#df['persons'] = df['persons'].astype('category')
#df['lug_boot'] = df['lug_boot'].astype('category')
#df['safety'] = df['safety'].astype('category')
df['class'] = df['class'].astype('category')
print(df.dtypes)
#cat_columns = df.select_dtypes(['category']).columns
#df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

df[:] = df[:].astype('category')
df[:] = df[:].apply(lambda x: x.cat.codes)
print(df.values.shape)

X = df.values[:, :6]
Y = df.values[:, 6]
print(X, '====', Y)
Nlbl = len(np.unique(Y)) # Number of classes 

print(X.ptp(0))
X_norm = (X - X.min(0)) / X.ptp(0)  # X.ptp - peak to peak range
#print(X_norm)

Y_1hot = pd.get_dummies(Y)  # one hot coding
#print(Y_1hot)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
import seaborn as sns

kmeans_model = KMeans(n_clusters=2, random_state=1).fit(X_norm)
labels_pred = kmeans_model.labels_
SC = metrics.silhouette_score(X_norm, labels_pred, metric='euclidean')  # inner metrics
#DB = metrics.davies_bouldin_score(X_norm, labels_pred)  # inner metrics

FMI = metrics.fowlkes_mallows_score(Y, labels_pred)  # external metrics

print('SC, DB, FMI = ', SC, DB, FMI)


################## Graph metrics
sse = []
fmi = []
dbi = []
k_list = range(1, 15)
for k in k_list:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(X_norm)
    labels_pred = km.labels_
    SC = metrics.silhouette_score(X_norm, labels_pred, metric='euclidean')
    DB = metrics.davies_bouldin_score(X_norm, labels_pred)  # inner metrics
    #print(k, SC)
    sse.append([k, SC])
    dbi.append([k, DB])
    fmi.append([k, metrics.fowlkes_mallows_score(Y, labels_pred)])

    
#oca_results_scale = pd.DataFrame({'Cluster': range(2,15), 'SSE': sse})
oca_results_scale = pd.DataFrame({'Cluster': range(1,15), 'FMI': fmi})
plt.figure(figsize=(12,6))
plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1], marker='o')
#plt.plot(pd.DataFrame(dbi)[0], pd.DataFrame(dbi)[1], marker='o')
#plt.plot(pd.DataFrame(fmi)[0], pd.DataFrame(fmi)[1], marker='o')
plt.title('Optimal Number of Clusters using Elbow Method (Scaled Data)')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.show()

'''
############### PCA for visualization
pca2 = PCA(n_components=2).fit(X_norm)
pca2d = pca2.transform(X_norm)
print(len(pca2d[:,0]), len(pca2d[:,1]))
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue=labels_pred, 
                #hue=Y, 
                palette='Set1',
                s=100, alpha=0.2).set_title('KMeans Clusters (4) Derived from Original Dataset', fontsize=15)
plt.legend()
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()
'''



