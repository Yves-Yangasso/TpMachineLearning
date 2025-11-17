# ========================================
# SCRIPT COMPLET : Exercices Régression Linéaire - Dataset Diabetes (Scikit-Learn)
# ========================================


# --- IMPORTS GLOBAUX ---
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
# ========================================

# --- EXERCICE 1 : Chargement et Exploration des Données (20 min) ---
print("=== EXERCICE 1 : Chargement et Exploration ===")
diabetes = load_diabetes()

# Affichage basique
print(f"Nombre d’échantillons : {diabetes.data.shape[0]}")  # 442
print(f"Nombre de features : {diabetes.data.shape[1]}")     # 10
print(f"Noms des features : {diabetes.feature_names}")     # ['age', 'sex', 'bmi', ...]
print("Description de la cible (extrait) :\n", diabetes.DESCR[:500] + "...")  # Mesure quantitative progression

# DataFrame Pandas
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
print("\n5 premières lignes :\n", df.head())
print("\nStatistiques descriptives :\n", df.describe())

# Réflexion : Pourquoi régression ? Cible continue (progression numérique 25-346), non catégorielle (vs. classif. binaire).
print("\n[RÉFLEXION EX1] : Dataset pour régression car cible quantitative (prédit valeur, pas classe).")
# ========================================

# --- EXERCICE 2 : Visualisation des Données (30 min) ---
print("\n=== EXERCICE 2 : Visualisation ===")

# Scatterplot BMI vs target
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='bmi', y='target')
sns.regplot(data=df, x='bmi', y='target', scatter=False, color='red')
plt.title('BMI vs. Progression du Diabète')
plt.show()

# Matrice de corrélation (heatmap)
plt.figure(figsize=(10,8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de Corrélation')
plt.show()
# Quelle feature la plus corrélée ? 'bmi' (~0.59)
print("[RÉFLEXION EX2 - Corrélée] : 'bmi' (0.59) est la plus corrélée à 'target'.")

# Histogramme cible
plt.figure(figsize=(8,6))
sns.histplot(data=df, x='target', kde=True)
plt.title('Distribution de la Progression du Diabète')
plt.show()
# Distribution normale ? Oui, approximativement (unimodale, symétrique).

# Réflexion : Relation linéaire modérée (visibles pour 'bmi') → Justifie régression linéaire (assume linéarité, simple).
print("[RÉFLEXION EX2] : Relation linéaire claire modérée ; justifie régr. lin. comme baseline interprétable.")
# ========================================

# --- EXERCICE 3 : Construction d’un Modèle de Régression Linéaire (40 min) ---
print("\n=== EXERCICE 3 : Modèle Régression Linéaire ===")
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

# Modèle multiple
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Évaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE sur test : {mse:.2f}")
print(f"R² sur test : {r2:.2f}")

# Coefficients
print("Coefficients :", dict(zip(diabetes.feature_names, model.coef_)))

# Impact BMI
bmi_idx = diabetes.feature_names.index('bmi')
print(f"Coefficient BMI : {model.coef_[bmi_idx]:.2f}")  # Impact positif fort

# Prédiction exemple
print(f"Prédiction échantillon 0 test : {y_pred[0]:.2f} (vrai : {y_test[0]:.2f})")

# Variante : Régression simple sur 'bmi'
X_bmi_train = X_train[:, bmi_idx].reshape(-1, 1)
X_bmi_test = X_test[:, bmi_idx].reshape(-1, 1)
model_simple = LinearRegression()
model_simple.fit(X_bmi_train, y_train)
y_pred_simple = model_simple.predict(X_bmi_test)
r2_simple = r2_score(y_test, y_pred_simple)
print(f"R² simple (BMI) : {r2_simple:.2f} (vs. multiple : {r2:.2f})")  # Multiple mieux

# Réflexion : R² >0.5 satisfaisant ? Modéré ici (0.45) ; risques overfitting si train>>test (ici faible, peu features).
print("[RÉFLEXION EX3] : R²=0.45 modéré (acceptable baseline) ; overfitting risque si variance haute, mais CV confirmera.")
# ========================================

# --- EXERCICE 4 : Amélioration et Évaluation Avancée (30 min) ---
print("\n=== EXERCICE 4 : Amélioration et Évaluation ===")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)
print(f"R² sur test scalé : {r2_scaled:.2f}")  # Inchangé (LR invariante)

# Cross-validation
scores = cross_val_score(LinearRegression(), diabetes.data, diabetes.target, cv=5, scoring='r2')
print(f"R² moyen CV : {scores.mean():.2f} (±{scores.std():.2f})")  # Stable ~0.45

# Scatter plot vraies vs prédites
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Vraies valeurs')
plt.ylabel('Prédites')
plt.title('Vraies vs Prédites (Test)')
plt.show()


print("\n=== SCRIPT TERMINÉ : Tous exercices couverts. Graphs générés. Préparez session 3 (Classification) ! ===")
