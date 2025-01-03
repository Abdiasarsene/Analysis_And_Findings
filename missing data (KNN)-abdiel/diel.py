# Importation des librairies
import pandas as pd
from sklearn.impute  import KNNImputer

# Importation de la base de données 
diel = pd.read_excel(r'C:\Users\HP ELITEBOOK 840 G6\Downloads\Baseeze.xlsx')

# Imputation de la base de données
voisi = KNNImputer(n_neighbors=3)
diel_data = pd.DataFrame(voisi.fit_transform(diel), columns=diel.columns)

# Affichage des données après imputation
diel_data

# Extraction de la base de données
diel_data.to_excel("diel_data.xlsx", index=False)

# Tronquage de la base de données
observation = 9000
data = diel.head(observation)