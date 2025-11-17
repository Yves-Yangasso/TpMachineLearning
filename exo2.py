import matplotlib.pyplot as plt
import seaborn as sns
import exo1

# Scatterplot BMI vs target avec ligne de régression
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='bmi', y='target')
sns.regplot(data=df, x='bmi', y='target', scatter=False, color='red')
plt.title('BMI vs. Progression du Diabète')
plt.show()

# Matrice de corrélation en heatmap
plt.figure(figsize=(10,8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de Corrélation')
plt.show()

# Histogramme de la cible
plt.figure(figsize=(8,6))
sns.histplot(data=df, x='target', kde=True)
plt.title('Distribution de la Progression du Diabète')
plt.show()