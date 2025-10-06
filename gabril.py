import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Carrega os dados
dados = pd.read_csv(r'C:\Users\guilh\OneDrive\Área de Trabalho\Trabalho-Machine-Learning\Steel_industry_data.csv')

# Define as features e rótulos
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

#Padronização
X_scaled = StandardScaler().fit_transform(X)

# PCA (3 componentes)
pca = PCA(n_components=3)
X_r = pca.fit_transform(X_scaled)

# Clustering com base nos dados PCA
clustering = AgglomerativeClustering(n_clusters=3)
y_ac = clustering.fit_predict(X_r)

# Normaliza apenas para plot 
x1 = (X_r[:,0] - X_r[:,0].min()) / (X_r[:,0].max() - X_r[:,0].min())
y1 = (X_r[:,1] - X_r[:,1].min()) / (X_r[:,1].max() - X_r[:,1].min())

# Cores e marcadores
col = ['r','b','g','m','y','c','k']
mar = ['o','^','s','x','+','*','8']
clabel = [col[i % len(col)] for i in y_ac]
mlabel = [mar[i % len(mar)] for i in y_ac]

# === Gráfico PCA ===
plt.figure(figsize=(10, 7))
for i in range(len(x1)):
    plt.scatter(x1[i], y1[i], marker=mlabel[i], s=25, c=clabel[i])

plt.xlabel('Componente 1 expressa ' + str("{:.2f}".format(100 * pca.explained_variance_ratio_[0])) + '% da variabilidade')
plt.ylabel('Componente 2 expressa ' + str("{:.2f}".format(100 * pca.explained_variance_ratio_[1])) + '% da variabilidade')
plt.title('PCA + Agglomerative Clustering - Steel Industry')
plt.grid(True)
plt.tight_layout()
plt.show()