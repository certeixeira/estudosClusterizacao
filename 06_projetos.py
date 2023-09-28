# IDENTIFICAÇÃO ANIMAIS ZOOLOGICO
# %%
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from itertools import combinations
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', palette='pastel')

import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('zoo.csv')
df
# %%
