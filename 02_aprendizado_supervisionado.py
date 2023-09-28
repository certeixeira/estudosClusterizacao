# APRENDIZAGEM SUPERVISIONADA
# %%

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
# %%
data = pd.read_csv('penguins_size.csv')
data
# %%
X, y = data.dropna().iloc[:, 2:-1], data.dropna().iloc[:,0]
# %%
X
# %%
le = LabelEncoder()
Y_ = le.fit_transform(y)
# %%
clf = DecisionTreeClassifier().fit(X, y)
# %%
plot_tree(clf)
# %%

r = export_text(clf, feature_names=list(X.columns))
print(r)
# %%

# APRENDIZADO NÃO SUPERVISIONADO

# %%
X
# %%
X_ = X.iloc[:, [0,2]].values
X_
# %%
plt.scatter(X_[:,0], X_[:,1], color='orange', edgecolors='black', s=15)
plt.xlabel('comp_bico_mm')
plt.ylabel('comp_asa_mm')
# %%
from sklearn.cluster import KMeans
# %%

kmeans = KMeans(n_clusters=3, random_state=3, n_init='auto').fit(X_)
labels = kmeans.labels_
# %%

# resultados kmeans
plt.scatter(X_[labels == 0, 0], X_[labels == 0, 1], color='blue', edgecolors='black', s=15)
plt.scatter(X_[labels == 1, 0], X_[labels == 1, 1], color='red', edgecolors='black', s=15)
plt.scatter(X_[labels == 2, 0], X_[labels == 2, 1], color='green', edgecolors='black', s=15)
plt.xlabel('comp_bico_mm')
plt.ylabel('comp_asa_mm')
# %%

# dados reais
plt.scatter(X_[Y_ == 0, 0], X_[Y_ == 0, 1], color='blue', edgecolors='black', s=15)
plt.scatter(X_[Y_ == 1, 0], X_[Y_ == 1, 1], color='red', edgecolors='black', s=15)
plt.scatter(X_[Y_ == 2, 0], X_[Y_ == 2, 1], color='green', edgecolors='black', s=15)
plt.xlabel('comp_bico_mm')
plt.ylabel('comp_asa_mm')
# %%
