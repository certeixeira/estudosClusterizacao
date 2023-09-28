# REPRESENTAÇÃO DE PROBLEMA
# %%

import pandas as pd
import seaborn as sns
sns.set_theme(style='darkgrid', palette='pastel')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
import warnings
warnings.filterwarnings('ignore')
# %%

data = pd.read_csv('penguins_size.csv')
data
# %%

data.tail()
# %%

# verificando o % de valores faltantes
temp = data.isna().sum() / (len(data)) * 100
temp
# %%
# podemos preencher os valores faltantes com a média ou a moda
# moda na coluna 'sex'
print(data['sex'].mode())
data['sex'].fillna(data['sex'].mode()[0])
# %%
# média na coluna 'flipper_length_mm'
print(data['flipper_length_mm'].mean())
# %%
sns.displot(data['flipper_length_mm'].values)
# %%

# no caso iremos apagar as linhas com dados Na
data = data.dropna()
# %%

# buscando os outliers
columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

for i in columns:
    sns.displot(data[i].values)\
        .set(title=i)
# %%
data = data[data['culmen_length_mm'] < 50]
# %%

# NORMALIZAÇÃO

# %%

X, y = data.dropna().iloc[:, 2: -1], data.dropna().iloc[:, 0]
# %%

# ajustando os valores entre 0 e 1
scaler = MinMaxScaler()
X_ = scaler.fit_transform(X)
X_.max()
X_
# %%

# ALGORITMOS DE AGRUPAMENTO

# %%

x = X_[:, [0, 2]]
x
# %%

kmeans = KMeans(n_clusters=3, random_state=3, n_init='auto').fit(x)
labels = kmeans.labels_
# %%

plt.scatter(x[labels == 0, 0], x[labels == 0, 1], color='blue', edgecolors='black', s=15)
plt.scatter(x[labels == 1, 0], x[labels == 1, 1], color='red', edgecolors='black', s=15)
plt.scatter(x[labels == 2, 0], x[labels == 2, 1], color='green', edgecolors='black', s=15)
plt.xlabel('comp_bico_mm')
plt.ylabel('comp_asa_mm')
# %%

# ALGORITMOS DE DENSIDADE

# %%

db = DBSCAN(eps=0.07, min_samples=10).fit(x)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_= list(labels).count(-1)

print('Número de clusters estimados: %d' % n_clusters_)
print('Número de pontos de ruídos estimados: %d' % n_noise_)
# %%
plt.scatter(x[labels == 0, 0], x[labels == 0, 1], color='blue', edgecolors='black', s=15)
plt.scatter(x[labels == 1, 0], x[labels == 1, 1], color='red', edgecolors='black', s=15)
plt.scatter(x[labels == -1, 0], x[labels == -1, 1], color='green', edgecolors='black', s=15)
plt.xlabel('comp_bico_mm')
plt.ylabel('comp_asa_mm')
# %%

# AGRUPAMENTO HIERÁRQUICO

# %%

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X_)
# %%

dendrogram(Z, truncate_mode='lastp')
# %%
