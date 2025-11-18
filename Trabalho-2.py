import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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

# Balanceamento (undersampling)
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_scaled, y)

print("Distribuição antes do balanceamento:", Counter(y))
print("Distribuição depois do balanceamento:", Counter(y_res))

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.20, random_state=42, stratify=y_res
)

# Modelo base + K-Fold
modelo = DecisionTreeClassifier(random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(modelo, X_train, y_train, cv=kfold, scoring='accuracy')
print("\nResultados de cada fold:", scores)
print("Acurácia média:", scores.mean())

# GridSearchCV para ajuste de hiperparâmetros
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=modelo,
    param_grid=param_grid,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nMelhores parâmetros (treino balanceado):")
print(grid_search.best_params_)
print("Melhor acurácia média obtida (validação, balanceado):", grid_search.best_score_)

# Repetir 30 execuções com o melhor modelo
melhor_modelo = grid_search.best_estimator_

acuracias, precisions, recalls, f1s = [], [], [], []

for i in range(30):
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_res, y_res, test_size=0.20, random_state=i, stratify=y_res
    )
    
    melhor_modelo.fit(X_train_i, y_train_i)
    y_pred = melhor_modelo.predict(X_test_i)
    
    acuracias.append(accuracy_score(y_test_i, y_pred))
    precisions.append(precision_score(y_test_i, y_pred, average='weighted', zero_division=0))
    recalls.append(recall_score(y_test_i, y_pred, average='weighted', zero_division=0))
    f1s.append(f1_score(y_test_i, y_pred, average='weighted', zero_division=0))

print("\nMÉDIAS DAS 30 EXECUÇÕES (modelo treinado com dados BALANCEADOS):")
print(f"Acurácia média: {np.mean(acuracias):.4f}")
print(f"Precisão média: {np.mean(precisions):.4f}")
print(f"Recall médio: {np.mean(recalls):.4f}")
print(f"F1-score médio: {np.mean(f1s):.4f}")

# Avaliar o melhor modelo no teste (balanceado)
y_test_pred = melhor_modelo.predict(X_test)

print("\nAvaliação do melhor modelo (dados balanceados):")
print("Acurácia:", accuracy_score(y_test, y_test_pred))
print("Precisão (weighted):", precision_score(y_test, y_test_pred, average='weighted', zero_division=0))
print("Recall (weighted):", recall_score(y_test, y_test_pred, average='weighted', zero_division=0))
print("F1-score (weighted):", f1_score(y_test, y_test_pred, average='weighted', zero_division=0))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))

# Comparação com base original (não balanceada)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

grid_search_orig = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1
)
grid_search_orig.fit(X_train_orig, y_train_orig)

print("\nMelhores parâmetros (dados originais):")
print(grid_search_orig.best_params_)
print("Melhor acurácia média (validação, original):", grid_search_orig.best_score_)

melhor_modelo_orig = grid_search_orig.best_estimator_

y_test_orig_pred = melhor_modelo_orig.predict(X_test_orig)

print("\nAvaliação do melhor modelo (dados originais):")
print("Acurácia:", accuracy_score(y_test_orig, y_test_orig_pred))
print("Precisão (weighted):", precision_score(y_test_orig, y_test_orig_pred, average='weighted', zero_division=0))
print("Recall (weighted):", recall_score(y_test_orig, y_test_orig_pred, average='weighted', zero_division=0))
print("F1-score (weighted):", f1_score(y_test_orig, y_test_orig_pred, average='weighted', zero_division=0))
print("\nClassification Report:\n", classification_report(y_test_orig, y_test_orig_pred, zero_division=0))

# Comparativo final
summary = pd.DataFrame({
    'Conjunto': ['Balanceado (modelo treinado balanceado)', 'Original (modelo treinado original)'],
    'Acurácia': [
        accuracy_score(y_test, y_test_pred),
        accuracy_score(y_test_orig, y_test_orig_pred)
    ],
    'Precisão_weighted': [
        precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        precision_score(y_test_orig, y_test_orig_pred, average='weighted', zero_division=0)
    ],
    'Recall_weighted': [
        recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        recall_score(y_test_orig, y_test_orig_pred, average='weighted', zero_division=0)
    ],
    'F1_weighted': [
        f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
        f1_score(y_test_orig, y_test_orig_pred, average='weighted', zero_division=0)
    ]
})

print("\nRESUMO COMPARATIVO:")
print(summary.to_string(index=False))
