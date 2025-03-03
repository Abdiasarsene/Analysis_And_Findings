# iMPORTATION DES BIBLIOTHEQUES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# GENERATION DES DONNEES ALEATOIRES
# Fixer la graine pour la reproductibilité
np.random.seed(42)

# Générer des données aléatoires
data, labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])
df #affichage des données aléatoires
plt.scatter(df['Feature1'], df['Feature2'])
plt.title('Données générées')
plt.show()

# APPLICATION DE L'ALGO DE K-MEANS
# Initialiser K-means avec 4 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(df)

# Ajouter les labels des clusters au DataFrame
df['Cluster'] = kmeans.labels_
df #Affichage des clusters

# VISUALISATION DES CLUSTERS
plt.scatter(df['Feature1'], df['Feature2'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Clusters et Centroids')
plt.legend()
plt.show()

# Autre raccourcis pour la visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Visualisation des clusters avec seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Feature1', y='Feature2', hue='Cluster', data=df, palette='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, color='red', label='Centroids')
plt.title('Clusters et Centroids')
plt.legend()
plt.show()

# INTERPRETATION DES RESULTATS
centroids = kmeans.cluster_centers_
print('Centroids des clusters:')
print(centroids)
