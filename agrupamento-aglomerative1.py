import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import ParameterGrid

# Carregar os dados
dados = pd.read_csv(r'C:\Users\guilh\OneDrive\Área de Trabalho\Trabalho-Machine-Learning\Steel_industry_data.csv')

# Definir features
feature_names = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM"
]
X = dados[feature_names].values

# Padronização dos dados
X_scaled = StandardScaler().fit_transform(X)

# PCA inicial para verificar componentes principais
pca_check = PCA(n_components=3)
X_pca_check = pca_check.fit_transform(X_scaled)
print("Variância explicada pelas 3 primeiras componentes principais:")
for i, var in enumerate(pca_check.explained_variance_ratio_):
    print(f"Componente {i+1}: {var*100:.2f}%")

# Grid Search para encontrar os melhores parâmetros do Agglomerative
param_grid = {
    "n_clusters": [2, 3, 4, 5],
    "linkage": ["ward", "complete", "average", "single"]
}

best_score = -1
best_param = None

for params in ParameterGrid(param_grid):
    try:
        clustering_temp = AgglomerativeClustering(**params)
        labels_temp = clustering_temp.fit_predict(X_scaled)
        score_temp = calinski_harabasz_score(X_scaled, labels_temp)
        if score_temp > best_score:
            best_score = score_temp
            best_param = params
    except Exception:
        continue

print("\nMelhores parâmetros encontrados:", best_param)
print(f"Melhor Calinski-Harabasz Score: {best_score:.3f}")

# Aplicar Agglomerative com os melhores parâmetros
clustering_best = AgglomerativeClustering(**best_param)
labels_best = clustering_best.fit_predict(X_scaled)

# PCA final apenas para visualização (2 componentes)
pca_vis = PCA(n_components=2)
X_pca_vis = pca_vis.fit_transform(X_scaled)

# Gráfico dos clusters
plt.figure(figsize=(10, 7))
col = ['r','b','g','m','y','c','k']
for i in range(len(X_pca_vis)):
    plt.scatter(X_pca_vis[i,0], X_pca_vis[i,1], color=col[labels_best[i] % len(col)], s=25)

plt.xlabel(f'Componente 1 ({pca_vis.explained_variance_ratio_[0]*100:.2f}% variância)')
plt.ylabel(f'Componente 2 ({pca_vis.explained_variance_ratio_[1]*100:.2f}% variância)')
plt.title('Agglomerative Clustering + PCA (visualização)')
plt.grid(True)
plt.tight_layout()
plt.show()
