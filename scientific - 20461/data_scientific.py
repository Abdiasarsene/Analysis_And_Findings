# Importation des librairies
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from googletrans import Translator
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors

# Importation de la base de données
data_scienti = pd.read_excel(r'D:\Projets\Projet IT\Projet Informatique\datascience\projects\databases\data_scientific.xlsx')
data_scienti

# Traduction des vairables en anglais
translation = Translator()

# Traduire les noms de colonnes 
translated_columns = {col: translation.translate(col, src='fr', dest='en').text for col in data_scienti.columns} 

# Renommer les colonnes avec les noms traduits 
data_scienti.rename(columns=translated_columns, inplace=True)
data_scienti

# Utilisation du KNN pour corriger les valeurs manquantes
imputer = KNNImputer(n_neighbors=3)
data_imputed = pd.DataFrame(imputer.fit_transform(data_scienti), columns=data_scienti.columns)

# Détection des outliers en utilisant l'IQR
Q1 = data_imputed.quantile(0.25)
Q3 = data_imputed.quantile(0.75)
IQR = Q3 - Q1

# Conditions pour identifier les outliers
is_outlier = (data_imputed < (Q1 - 1.5 * IQR)) | (data_imputed > (Q3 + 1.5 * IQR))

# Initialiser le modèle KNN
nbrs = NearestNeighbors(n_neighbors=2)
nbrs.fit(data_imputed)

# Remplacer les outliers par la moyenne de leurs k voisins les plus proches
for col in data_imputed.columns:
    for idx in data_imputed[col][is_outlier[col]].index:
        distances, indices = nbrs.kneighbors([data_imputed.loc[idx, :]])
        knn_mean = data_imputed.loc[indices[0]].mean()
        data_imputed.at[idx, col] = knn_mean[col]

# Visualisation de la base de données
msno.bar(data_imputed, color='purple')

# Normalisation des données
scaler = MinMaxScaler()
data_normalized = pd.DataFrame( data = scaler.fit_transform(data_imputed),columns= data_imputed.columns)

# Concaténation des bases de données
data_normalized = data_normalized.drop(columns=['Years'])
files = 'D:\Projets\Projet IT\Projet Informatique\datascience\projects\databases\extract_temper.xlsx'
extract_data = pd.read_excel(files)
extract_data.drop(columns=['Unnamed: 0', 'Température moyenne'])

extract_data.rename(columns={"Années":'Years'}, inplace=True)

scientifc_data = pd.concat([extract_data, data_normalized], axis=1)
scientifc_data

scientifc_data.drop(columns=['Unnamed: 0','Température moyenne'])

