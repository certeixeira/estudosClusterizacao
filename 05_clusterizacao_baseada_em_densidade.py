# ALGORITMO DBSCAN
# %%    
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid', palette='pastel')
# %%
# clusters

centers = [(-4, -4), (-4, 4), (4, -4), (4, 4)]
cluster_std = 1.4

X, y= make_blobs(n_samples=600, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)

plt.scatter(X[:, 0], X[:, 1], color='blue', edgecolors='black', s=15)
# %%

db = DBSCAN(eps=1, min_samples=3).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('numero de clusters: %d' % n_clusters_)
print('numero de ruidos: %d' % n_noise_)
# %%

def  plot_db(X, labels):
    df_db = pd.DataFrame(X)
    df_db['cluster'] = labels
    
    sns.scatterplot(data=df_db,
                    x=0,
                    y=1,
                    hue='cluster',
                    palette='deep')
# %%
plot_db(X, labels)
# %%

# CLUSTERS NATURAIS

# %%

X, y = make_moons(600, noise=0.07, random_state=1)
plt.scatter(X[:, 0], X[:, 1], color='blue', edgecolors='black', alpha=.6, s=15)
# %%

db = DBSCAN(eps=.15, min_samples=4).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('numero de clusters: %d' % n_clusters_)
print('numero de ruidos: %d' % n_noise_)
# %%

plot_db(X, labels)
# %%

k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
labels = kmeans.labels_

plot_db(X, labels)

# %%
