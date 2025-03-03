# IMPORTATION DES LIBRAIRIES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# GENERATION DES DONNEES ALEATOIRES
# Fixer la graine pour la reproductibilité
np.random.seed(42)

# Générer des données aléatoires
data = np.random.rand(100, 5)
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
df #Affichage de la base de données

# STANDARDISATION DES DONNEES
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_data#Affichage des données standardisées

# EFFECTUER ACP
# Initialiser l'ACP pour réduire à 2 composantes principales
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Créer un DataFrame avec les composantes principales
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pca_df #Affichage de l'ACP réalisée

# VISUALISATION DES DONNEES
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], c='blue', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 Component PCA')
plt.show()

# INTERPRETER DES RESULTATS
explained_variance = pca.explained_variance_ratio_
print(f'Variance expliquée par la première composante principale: {explained_variance[0]:.2f}')
print(f'Variance expliquée par la deuxième composante principale: {explained_variance[1]:.2f}')
