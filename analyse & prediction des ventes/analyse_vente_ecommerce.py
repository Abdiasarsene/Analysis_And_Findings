# Importation des bibliothèques
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Importation de la base de données
ecommerce = pd.read_excel(r"D:\Projets\Projet IT\Projet Datascience\reelproject\analyse & prediction des ventes\synthetic_sales_data.xlsx")

# Vérification des valeurs manquantes
ecommerce.isnull().sum()

# Supression des doublons
ecommerce = ecommerce.drop_duplicates()

# Conversion des dates en datetime
ecommerce['Date'] = pd.to_datetime(ecommerce['Date'])

# Extraction du mois et du trimestre
ecommerce['Mois'] = ecommerce['Date'].dt.month
ecommerce['Trimestre'] = ecommerce['Date'].dt.to_period('Q')

# Statistique descriptives
statistic =ecommerce.select_dtypes(exclude=['object','datetime64','period[Q-DEC]']).describe()

# Total des ventes
total_ventes = ecommerce['Total_Vente'].sum()
print('Totaux des ventes =',total_ventes)

# Ventes par catégories de produits
ventes_par_categorie = ecommerce.groupby('Product_Category')['Total_Vente'].sum()
print('Les ventes réalisées par catégorie de produits')
ventes_par_categorie

# Ventes par trimestre
ventes_par_trimestre = ecommerce.groupby('Trimestre')['Total_Vente'].sum()
print('Les ventes trimestrielles')
ventes_par_trimestre

# Ventes par saison
ventes_par_saison = ecommerce.groupby('Saison')['Total_Vente'].sum()
print('Les ventes saisonnières')
ventes_par_saison

# Ventes par mois
ventes_par_mois = ecommerce.groupby('Mois')['Total_Vente'].sum()
print('Les ventes mensuelles')
ventes_par_mois

# Ventes par région
ventes_par_region = ecommerce.groupby('Region')['Total_Vente'].sum()
print('Les ventes régionales')
ventes_par_region

# Ventes par météo
ventes_par_meteo = ecommerce.groupby('Météo')['Total_Vente'].sum()
print('Les ventes météorologiques')
ventes_par_meteo

# Courbe des ventes au cours du temps
sns.set( style="whitegrid", palette="deep", font_scale=1.1)
ventes_par_date = ecommerce.groupby('Date')['Total_Vente'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=ventes_par_date, x='Date', y='Total_Vente', color='royalblue')
plt.title("Tendance des ventes au fil du temps")
plt.xlabel("Date")
plt.ylabel("Montant des ventes")
plt.show()

# Répartition des ventes par catégorie
sns.set(style='white', palette='muted',font_scale=1)
ventes_par_categorie = ecommerce.groupby('Product_Category')['Total_Vente'].sum().reset_index()
plt.figure(figsize=(12,6))
sns.barplot(data=ventes_par_categorie,x='Product_Category', y='Total_Vente', palette='viridis')
plt.title("Répartition des ventes par catégorie de produits")
plt.xlabel("Catégorie de produits")
plt.ylabel("Total des ventes")
plt.show()

# Tendances des ventes par trimestre
sns.set(style='whitegrid', palette="deep", font_scale=1.1)
ventes_par_trimestre = ecommerce.groupby('Trimestre')['Total_Vente'].sum().reset_index()
ventes_par_trimestre['Trimestre']= ventes_par_trimestre['Trimestre'].astype(str)
plt.figure(figsize=(12, 6))
sns.lineplot(data=ventes_par_trimestre, x='Trimestre',y='Total_Vente', color='purple', marker='o')
plt.title('Tendance des ventes trimestrielles')
plt.xlabel('Trimestre')
plt.ylabel('Total des ventes')
plt.show()

# Heatmap des ventes par mois et catégorie
sns.set(style='white', palette='deep', font_scale=1.1)
pivot_table = ecommerce.pivot_table(values='Total_Vente', index='Mois', columns='Product_Category', aggfunc='sum')
plt.figure(figsize=(12,6))
sns.heatmap(pivot_table, annot=True, cmap="Blues")
plt.title("Heatmap des ventes par mois et catégorie")
plt.show()

# Identification des meilleurs produits
top_produits = ecommerce.groupby('Product_Category')['Total_Vente'].sum().sort_values(ascending=False).head(3)
print('Nos meilleurs produits après analyse')
print(top_produits)

# o	Identification des pics de vente liés aux promotions ou jours fériés.
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# 1. Agréger les ventes par date pour identifier les pics
ventes_par_date = ecommerce.groupby('Date')['Total_Vente'].sum().reset_index()

# 2. Créer des sous-ensembles pour Promotions et Jours fériés
ventes_promotion = ecommerce[ecommerce['Promotion_Event'] == True].groupby('Date')['Total_Vente'].sum().reset_index()
ventes_ferie = ecommerce[ecommerce['Vacances_Fériées'] == True].groupby('Date')['Total_Vente'].sum().reset_index()

# 3. Visualisation des ventes avec des pics marqués
plt.figure(figsize=(14, 6))

# Courbe des ventes globales
sns.lineplot(data=ventes_par_date, x='Date', y='Total_Vente', label="Ventes Globales", color='gray')

# Points pour les événements promotionnels
sns.scatterplot(data=ventes_promotion, x='Date', y='Total_Vente', color='purple', s=100, label='Événements Promotionnels')

# Points pour les jours fériés
sns.scatterplot(data=ventes_ferie, x='Date', y='Total_Vente', color='red', s=100, label='Jours Fériés')

# Ajout de titres et légendes
plt.title("Identification des pics de vente liés aux promotions et jours fériés")
plt.xlabel("Date")
plt.ylabel("Total des ventes")
plt.legend()
plt.xticks(rotation=45)
plt.show()

from statsmodels.tsa.stattools import adfuller

# Effectuer le test de Dickey-Fuller augmenté
result = adfuller(ventes_par_mois)

# Affichage des résultats avec interprétation
print("Résultats du test ADF:")
print(f"Statistique de test : {result[0]:.4f}")
print(f"p-value : {result[1]:.4f}")
print(f"Valeurs critiques :")
for key, value in result[4].items():
    print(f"   {key}: {value:.4f}")

# Interprétation
if result[1] < 0.05:
    print("La série est stationnaire (p-value < 0.05).")
else:
    print("La série n'est pas stationnaire (p-value >= 0.05). Considérez une transformation.")
    serie_stationnaire = ventes_par_mois.diff().dropna()  # Transformation par différenciation

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(ventes_par_mois, label="Original")
plt.plot(ventes_par_mois.diff(), label="Différenciée")
plt.legend()
plt.title("Série originale vs différenciée")
plt.show()

# Tracer les ACF et PACF
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# ACF (Autocorrélation)
plot_acf(serie_stationnaire, ax=axes[0], title="ACF (Autocorrélation)")

# PACF (Autocorrélation partielle)
plot_pacf(serie_stationnaire, ax=axes[1], title="PACF (Autocorrélation partielle)", method='ywm')

# Ajustement de la mise en page et affichage
plt.tight_layout()
plt.show()

# Entraînement du modèle ARIMA
model = ARIMA(ventes_par_mois, order=(1, 1, 1))  # p=1, d=1, q=1 comme point de départ
model_fit = model.fit()

# Résumé du modèle
print(model_fit.summary())

# Prévisions
forecast = model_fit.forecast(steps=12)
print(f"Prévisions pour les 12 prochains mois :\n{forecast}")

# Entraînement du modèle ARIMA avec des paramètres
model = ARIMA(ventes_par_mois, order=(2, 1, 2))  # p=1, d=1, q=1 comme point de départ
model_fit = model.fit()

# Résumé du modèle
print(model_fit.summary())

# Prévisions
forecast = model_fit.forecast(steps=12)
print(f"Prévisions pour les 12 prochains mois :\n{forecast}")

# Reconversion des dates en datetime excluant  "Q" dans l'écriture du trimestre
# Conversion des dates en datetime
ecommerce['Date'] = pd.to_datetime(ecommerce['Date'])

# Extraction du mois et du trimestre
ecommerce['Mois'] = ecommerce['Date'].dt.month

# Extraction du trimestre sous forme numérique (1, 2, 3, 4)
ecommerce['Trimestre'] = ecommerce['Date'].dt.quarter

# Si vous voulez un format 'année + trimestre' (par exemple '2025Q1')
# ecommerce['Trimestre'] = ecommerce['Date'].dt.to_period('Q').astype(str)

# Entraînement du modèle de Machine Learning, la forêt aléatoire et la prévision des ventes des 12 prochains mois
# 1. Load and preprocess the data
# Convert 'Date' to datetime (ensure it's not in a Period format)
ecommerce['Date'] = pd.to_datetime(ecommerce['Date'], errors='coerce')

# Extract 'Mois' and 'Année' (ensure these are numeric)
ecommerce['Mois'] = ecommerce['Date'].dt.month.astype(int)
ecommerce['Année'] = ecommerce['Date'].dt.year.astype(int)

# Encode categorical variables ('Saison', 'Region', 'Météo')
categorical_columns = ['Saison', 'Region', 'Météo']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_vars = pd.DataFrame(
    encoder.fit_transform(ecommerce[categorical_columns]),
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Concatenate encoded variables with the dataset and drop original categorical columns
ecommerce_cleaned = pd.concat([ecommerce.reset_index(drop=True), encoded_vars], axis=1)
ecommerce_cleaned = ecommerce_cleaned.drop(columns=categorical_columns + ['Date'])
ecommerce_cleaned = round(ecommerce_cleaned)
# Ensure 'Mois' and 'Année' are numeric
ecommerce_cleaned['Mois'] = ecommerce_cleaned['Mois'].astype(int)
ecommerce_cleaned['Année'] = ecommerce_cleaned['Année'].astype(int)

# 2. Prepare data for the model
X = ecommerce_cleaned.drop(columns=['Total_Vente','Product_ID','Customer_ID','Product_Category','Customer_Age_Group','Mode_d_Achat'])
y = ecommerce_cleaned['Total_Vente']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")

# 4. Generate predictions for the next 12 months
# Récupération des colonnes utilisées pour l'entraînement
required_columns = X_train.columns  # Colonnes utilisées pour entraîner le modèle

# Créez future_data avec les mêmes colonnes
future_data = pd.DataFrame({
    'Quantité_Vendue': [0] * 12,  # Par défaut à 0
    'Prix_Unitaire': [0] * 12,     # Par défaut à 0
    'Discount': [0] * 12,          # Par défaut à 0
    'Promotion_Event': [0] * 12,   # Par défaut à 0
    'Publicité_Spend': [0] * 12,   # Par défaut à 0
    'Vacances_Fériées': [0] * 12,  # Par défaut à 0
    'Mois': list(range(1, 13)),    # Mois de 1 à 12
    'Trimestre': [1] * 12,         # Trimestre (par exemple, 1 pour le premier trimestre)
    'Année': [2025] * 12,          # Année 2025 pour les prédictions futures
    'Saison_Hiver': [0] * 12,      # Par défaut à 0 pour l'encodage
    'Saison_Printemps': [0] * 12,  # Par défaut à 0 pour l'encodage
    'Saison_Été': [0] * 12,        # Par défaut à 0 pour l'encodage
    'Region_Nord': [0] * 12,       # Par défaut à 0 pour l'encodage
    'Region_Ouest': [0] * 12,      # Par défaut à 0 pour l'encodage
    'Region_Sud': [0] * 12,        # Par défaut à 0 pour l'encodage
    'Météo_Neigeux': [0] * 12,     # Par défaut à 0 pour l'encodage
    'Météo_Nuageux': [0] * 12,     # Par défaut à 0 pour l'encodage
    'Météo_Pluvieux': [0] * 12     # Par défaut à 0 pour l'encodage
})

# Réorganiser les colonnes dans le même ordre que dans X_train
future_data = future_data[required_columns]

# Prédire les ventes pour les 12 prochains mois
future_sales = rf_model.predict(future_data)
print("Prévisions des ventes pour les 12 prochains mois :")
print(future_sales)

import matplotlib.pyplot as plt

# Afficher les prédictions graphiquement
months = list(range(1, 13))  # Mois de 1 à 12

plt.figure(figsize=(10, 6))
plt.plot(months, future_sales, marker='o', linestyle='-', color='b', label="Ventes prévues")
plt.title("Prévisions des ventes pour les 12 prochains mois", fontsize=16)
plt.xlabel("Mois", fontsize=14)
plt.ylabel("Ventes prévues", fontsize=14)
plt.xticks(months)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()