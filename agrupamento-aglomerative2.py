import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import ParameterGrid

# Carregar os dados
dados = pd.read_csv(r'C:\Users\guilh\OneDrive\Área de Trabalho\Trabalho-Machine-Learning\Steel_industry_data.csv')

# Definir features e rótulos
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
y = dados["Load_Type"].values

# Padronização dos dados
X_scaled = StandardScaler().fit_transform(X)

# Seleção de características (SelectKBest)
selector = SelectKBest(score_func=f_classif, k=3) 
X_selected = selector.fit_transform(X_scaled, y)
selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
print("Características selecionadas:", selected_features)

# Grid Search para encontrar melhores parâmetros do Agglomerative
param_grid = {
    "n_clusters": [2, 3, 4, 5],
    "linkage": ["ward", "complete", "average", "single"]
}

best_score = -1
best_param = None

for params in ParameterGrid(param_grid):
    try:
        clustering_temp = AgglomerativeClustering(**params)
        labels_temp = clustering_temp.fit_predict(X_selected)
        score_temp = calinski_harabasz_score(X_selected, labels_temp)
        if score_temp > best_score:
            best_score = score_temp
            best_param = params
    except Exception:
        continue

print("\nMelhores parâmetros encontrados:", best_param)
print(f"Melhor Calinski-Harabasz Score: {best_score:.3f}")

# 6. Aplicar Agglomerative 
clustering_best = AgglomerativeClustering(**best_param)
labels_best = clustering_best.fit_predict(X_selected)

# 7. PCA 
pca_vis = PCA(n_components=2)
X_pca_vis = pca_vis.fit_transform(X_selected)

# 8. Gráfico dos clusters
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
