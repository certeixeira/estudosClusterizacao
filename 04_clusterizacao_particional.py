# IMPLEMENTAÇÃO DO ALGORITMO KMEANS EM PYTHON
# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
sns.set_theme(style='darkgrid', palette='pastel')
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# %%

data = pd.read_csv('penguins_size.csv')
data.head()
# %%
data = data.dropna()
# %%
# excluindo outliers
data = data[data.culmen_length_mm < 50]
# %%
X, y = data.iloc[:, 2:-1], data.iloc[:, 0]
# %%
scaler = MinMaxScaler()
# %%
X_ = scaler.fit_transform(X)
X_
# %%
x = X_[:, [0, 2]]
# %%
def k_means(data, k , threshold=2):
    n = np.size(data, 0)
    cluster_centers = np.random.choice(range(0, n), k)
    clustering = np.random.randint(0, k, n)
    cluster_means = data[cluster_centers]
    old_clustering = np.zeros(n)
    
    while np.sum(clustering != old_clustering) > threshold:
        old_clustering = clustering
        
        cluster_distance = np.zeros((n, k))
        for cluster in range(k):
            cluster_distance[:, cluster] = np.sum(np.sqrt((data - cluster_means[cluster]) ** 2), 1)
        clustering = np.argmin(cluster_distance, 1)
        
        cluster_means = np.array([np.mean(data[clustering == c], 0) for c in range(k)])
    
    return cluster_means, clustering
# %%
centers, labels = k_means(x, 3)
# %%
plt.scatter(x[labels == 0, 0], x[labels == 0, 1], color='blue', edgecolors='black', s=15)
plt.scatter(x[labels == 1, 0], x[labels == 1, 1], color='red', edgecolors='black', s=15)
plt.scatter(x[labels == 2, 0], x[labels == 2, 1], color='green', edgecolors='black', s=15)
plt.scatter(centers[:,0], centers[:,1], color='orange', marker='X')
plt.xlabel('comp_bico_mm')
plt.ylabel('comp_asa_mm')
# %%
