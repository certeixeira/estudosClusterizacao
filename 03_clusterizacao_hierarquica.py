# COMO FUNCIONA CLUSTERIZAÇÃO HERÁRQUICA
# %%
import pandas as pd
import seaborn as sns
sns.set_theme(style='darkgrid', palette='pastel')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
# %%

df = pd.read_csv('countries of the world.csv')
paises = df.Country.tolist()
print(paises)
# %%

paises = ['Argentina', 'Angola', 'Brazil', 'Japan', 'Chile']
mini_df = df[df.Country.str.strip().isin(paises)]
mini_df.columns


# %%
# transformando as colunas em float
columns_to_transform = ['Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
       'Net migration', 'Infant mortality (per 1000 births)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)',
       'Crops (%)', 'Other (%)', 'Birthrate', 'Deathrate',
       'Agriculture', 'Industry', 'Service']

for column in columns_to_transform:
    mini_df[column] = mini_df[column].str.replace(',', '.').astype(float)
    

mini_df.head(226)

# %%

# %%
X = mini_df.iloc[:, 4:10].dropna().copy()
Z = linkage(X)
# %%

plt.figure(figsize=(7,5))
dendrogram(Z, truncate_mode='lastp', labels=list(mini_df.Country)
           , orientation='top', leaf_font_size=10, leaf_rotation=0)
# %%

# DENDROGRAMA

# %%

country = ['Angola', 'Argentina', 'Australia', 'Brazil', 'Chile', 'Japan', 'Morocco', 'Senegal', 'Korea, South']
    
mini_df = df[df.Country.str.strip().isin(country)].copy()

columns_to_transform = ['Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
       'Net migration', 'Infant mortality (per 1000 births)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)',
       'Crops (%)', 'Other (%)', 'Birthrate', 'Deathrate',
       'Agriculture', 'Industry', 'Service']

for column in columns_to_transform:
    mini_df[column] = mini_df[column].str.replace(',', '.').astype(float)

X = mini_df.iloc[:, 4:10].dropna().copy()
X
# %%

Z = linkage(X, method='ward')

plt.figure(figsize=(7,5))
dendrogram(Z, labels=list(mini_df.Country),
           orientation='top', leaf_font_size=10, leaf_rotation=60)
# %%
