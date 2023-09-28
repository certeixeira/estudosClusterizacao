# INTRODUÇÃO A CLUSTERIZAÇÃO
# %%
# base de testes para utilizar 'make_blobs'
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid', palette='pastel')
# %%

# configura os centros dos grupos
centers = [(-5, -5), (-3, -3), (1, 1), (5, 5)]
# o quão espalhados estão os grupos (quanto menor, mais próximo estão)
clusters_std = [0.9, .6, 1, .8]

# base de dados com 200 exemplos
X, y = make_blobs(n_samples=200, cluster_std=clusters_std, centers=centers, n_features=2, random_state=1)
plt.scatter(X[:,0], X[:,1], color='blue', edgecolors='black', s=15)
# %%
# separando os grupospor cores para melhor visualização dos grupos
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolors='black', s=15)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolors='black', s=15)
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='green', edgecolors='black', s=15)
plt.scatter(X[y == 3, 0], X[y == 3, 1], color='orange', edgecolors='black', s=15)
# %%
