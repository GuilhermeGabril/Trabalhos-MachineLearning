import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif  # Exemplo para seleção de características
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid


# 1. Carregar os dados
dados = pd.read_csv(r'C:\Users\guilh\OneDrive\Área de Trabalho\Trabalho-Machine-Learning\Steel_industry_data.csv')

# 2. Definir features e rótulos
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
target_names = np.unique(y)

print("Características:", feature_names)
print("Classes:", target_names)

# 3. Padronização dos dados
X_scaled = StandardScaler().fit_transform(X)


# 4. PCA: redução para 3 componentes principais
pca = PCA(n_components=3)
X_r = pca.fit_transform(X_scaled)


# 5. Análise das características que mais influenciam PCA (loadings)
print("Coeficientes das 3 primeiras componentes principais:")
loadings = pca.components_.T  # Transposta para facilitar leitura (features x componentes)
for i, feature in enumerate(feature_names):
    print(f"{feature}: Componente 1 = {loadings[i, 0]:.4f}, Componente 2 = {loadings[i, 1]:.4f}, Componente 3 = {loadings[i, 2]:.4f}")


# 6. Seleção de características (exemplo com SelectKBest)
selector = SelectKBest(score_func=f_classif, k=3)  # escolha o valor k conforme desejar
X_selected = selector.fit_transform(X_scaled, y)
selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
print("Características selecionadas:", selected_features)


# 7. Aplicar clustering antes e após seleção de características
# Clustering na base original (com PCA)
clustering_original = AgglomerativeClustering(n_clusters=3)
labels_original = clustering_original.fit_predict(X_r)

# Fazer PCA na base selecionada para visualizar e clusterizar
pca_selected = PCA(n_components=3)
X_selected_r = pca_selected.fit_transform(X_selected)
clustering_selected = AgglomerativeClustering(n_clusters=3)
labels_selected = clustering_selected.fit_predict(X_selected_r)


# 8. Avaliar clusters usando critérios de validação
def avaliar_clusters(X, labels):
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return sil, ch, db

sil_o, ch_o, db_o = avaliar_clusters(X_r, labels_original)
sil_s, ch_s, db_s = avaliar_clusters(X_selected_r, labels_selected)

print(f"Clusters base original - Silhouette: {sil_o:.3f}, Calinski: {ch_o:.3f}, Davies-Bouldin: {db_o:.3f}")
print(f"Clusters base selecionada - Silhouette: {sil_s:.3f}, Calinski: {ch_s:.3f}, Davies-Bouldin: {db_s:.3f}")


# 9. Grid Search para encontrar melhores parâmetros do clustering (exemplo Agglomerative)
param_grid = {
    "n_clusters": [2, 3, 4, 5],
    "linkage": ["ward", "complete", "average", "single"]
}
best_score = -np.inf
best_param = None
lista_parametros_testados = []

for params in ParameterGrid(param_grid):
    clustering_temp = AgglomerativeClustering(**params)
    labels_temp = clustering_temp.fit_predict(X_r)
    score_temp = silhouette_score(X_r, labels_temp)
    lista_parametros_testados.append((params, score_temp))
    if score_temp > best_score:
        best_score = score_temp
        best_param = params

print("Parâmetros testados e seus scores:")
for p, s in lista_parametros_testados:
    print(p, f"Silhouette: {s:.3f}")

print("Melhores parâmetros encontrados:", best_param)
print(f"Melhor valor do critério de validação Silhouette: {best_score:.3f}")


# 10. Visualização dos clusters usando PCA
x1 = (X_r[:,0] - X_r[:,0].min()) / (X_r[:,0].max() - X_r[:,0].min())
y1 = (X_r[:,1] - X_r[:,1].min()) / (X_r[:,1].max() - X_r[:,1].min())

col = ['r','b','g','m','y','c','k']
mar = ['o','^','s','x','+','*','8']
clabel = [col[i % len(col)] for i in labels_original]
mlabel = [mar[i % len(mar)] for i in labels_original]

plt.figure(figsize=(10, 7))
for i in range(len(x1)):
    plt.scatter(x1[i], y1[i], marker=mlabel[i], s=25, c=clabel[i])

plt.xlabel('Componente 1 expressa ' + str("{:.2f}".format(100 * pca.explained_variance_ratio_[0])) + '% da variabilidade')
plt.ylabel('Componente 2 expressa ' + str("{:.2f}".format(100 * pca.explained_variance_ratio_[1])) + '% da variabilidade')
plt.title('PCA + Agglomerative Clustering - Steel Industry')
plt.grid(True)
plt.tight_layout()
plt.show()
