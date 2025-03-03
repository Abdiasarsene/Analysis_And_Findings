import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Générer des données aléatoires
np.random.seed(42)
data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])

# Visualiser les données générées
plt.scatter(df['Feature1'], df['Feature2'])
plt.title('Données générées')
plt.show()

# Standardiser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Appliquer l'ACP
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

# Appliquer K-means sur les données réduites
kmeans = KMeans(n_clusters=4)
kmeans.fit(pca_df)
pca_df['Cluster'] = kmeans.labels_

# Visualiser les clusters
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], c=pca_df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clusters après ACP')
plt.legend()
plt.show()

# Examiner les centres des clusters
centroids = kmeans.cluster_centers_
print('Centroids des clusters:')
print(centroids)
