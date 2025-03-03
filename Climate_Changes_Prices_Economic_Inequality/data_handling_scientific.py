# Importation des librairies
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importation de la base de données
data_scienti = pd.read_excel(r'C:\Users\HP ELITEBOOK 840 G6\Documents\Copie de BASE_DONNEES_CONSTITUEES(1).xlsx')
data_scienti

# Supression de variables inutiles
data_scienti.columns
data_scienti =data_scienti.drop(columns= ['Années.1','Années.2','Années.3','Années.4','Années.5','Années.6','Années.7','Années.8'])
data_scienti = data_scienti.drop(columns=['Années.1'])

# Visualisation graphique des données manquantes
msno.bar(data_scienti, color='green')

#  Tronquage de la base de données
data_observation = 100
data_scienti= data_scienti.head(data_observation)
data_scienti = data_scienti.drop(columns=['Années', 'Température moyenne'])
data_scienti

# Extraction de la variable "Température moyenne"
data_scienti.to_excel('complete_data.xlsx', index=False)

# Filtrer les années entre 1921 et 2022 inclusivement sur la variable Température moyenne
data_filtered = data_scienti[(data_scienti['Années'] >= 1921) & (data_scienti['Années'] <= 2022)] 
data_result = data_filtered[['Années', 'Température moyenne']]
data_result

# Extraction de la variable "Température moyenne"
data_result.to_excel('extract_temper.xlsx', index=False)

"""Concaténation pour créer une nouvelle variable """
# Importation des deux bases de données à concaténer
file_path = 'D:\Projets\Projet IT\Projet Informatique\datascience\projects\complete_data.xlsx'
complete_data = pd.read_excel(file_path)
complete_data

files = 'D:\Projets\Projet IT\Projet Informatique\datascience\projects\extract_temper.xlsx'
extract_data = pd.read_excel(files)
extract_data.drop(['Unnamed: 0'])
extract_data

# Concaténation
data_concat = pd.concat([extract_data,complete_data ], axis=1)
data_concat

# Supprssion de la variable "Unnamed: 0"
data_concat = data_concat.drop(columns=['Unnamed: 0'])

# Extraction de la base de données
data_concat.to_excel('concat_data.xlsx', index=False)

# Visualisation de la base de données
msno.bar(data_concat, color='green')

# Supression de certaines lignes
ligne_a_supprimer =[101]
data_concat = data_concat.drop(ligne_a_supprimer, inplace=True)
data_concat