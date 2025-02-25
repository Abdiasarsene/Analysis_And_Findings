import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Exemple d'importation des données
data = pd.read_csv("sales_data.csv")  # Remplacez par le nom de votre fichier
print(data.head())

# Vérification des valeurs manquantes
print(data.isnull().sum())

# Suppression des doublons
data = data.drop_duplicates()

# Conversion des dates en format datetime
data['Date'] = pd.to_datetime(data['Date'])

# Exemple : Extraire le mois ou le trimestre
data['Mois'] = data['Date'].dt.month
data['Trimestre'] = data['Date'].dt.to_period("Q")

# Statistiques descriptives
print(data.describe())

# Total des ventes
total_ventes = data['Montant'].sum()
print(f"Total des ventes : {total_ventes}")

# Ventes par catégorie de produit
ventes_par_categorie = data.groupby('Catégorie')['Montant'].sum()
print(ventes_par_categorie)

# Courbe des ventes au fil du temps
plt.figure(figsize=(10, 6))
data.groupby('Date')['Montant'].sum().plot()
plt.title("Tendance des ventes au fil du temps")
plt.xlabel("Date")
plt.ylabel("Montant des ventes")
plt.show()

# Répartition des ventes par catégorie
plt.figure(figsize=(8, 6))
sns.barplot(x=ventes_par_categorie.index, y=ventes_par_categorie.values)
plt.title("Ventes par catégorie de produit")
plt.xticks(rotation=45)
plt.show()

# Heatmap des ventes par mois et catégorie
pivot_table = data.pivot_table(values='Montant', index='Mois', columns='Catégorie', aggfunc='sum')
sns.heatmap(pivot_table, annot=True, cmap="Blues")
plt.title("Heatmap des ventes par mois et catégorie")
plt.show()

# Identification des produits performants
top_produits = data.groupby('Produit')['Montant'].sum().sort_values(ascending=False).head(5)
print(top_produits)

# Analyse  des ventes par région  ou par client
ventes_par_region = data.groupby('Région')['Montant'].sum()
print(ventes_par_region)


# Préparation des données pour la prédiction des ventes

# Assurez-vous que la colonne 'Date' est en format datetime
data['Date'] = pd.to_datetime(data['Date'])

# Grouper les données par mois pour avoir une série temporelle mensuelle des ventes
monthly_sales = data.groupby(data['Date'].dt.to_period('M'))['Montant'].sum()

# Convertir en format de série temporelle
monthly_sales = monthly_sales.to_timestamp()

# Visualiser les données historiques
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales)
plt.title('Ventes mensuelles historiques')
plt.xlabel('Date')
plt.ylabel('Montant des ventes')
plt.show()

# Stationnarisation des données
from statsmodels.tsa.stattools import adfuller

result = adfuller(monthly_sales)
print(f"Statistique de test : {result[0]}")
print(f"p-value : {result[1]}")

# Différenciation si la stationnarisation échoue
monthly_sales_diff = monthly_sales.diff().dropna()

"""Identification des paramètres du modèle ARIMA
Il faut maintenant identifier les paramètres de l'ARIMA, à savoir p, d, et q :

p : le nombre de périodes de l'autoregression
d : le nombre de différenciations nécessaires
q : le nombre de périodes de la moyenne mobile
Vous pouvez utiliser l'outil ACF (AutoCorrelation Function) et PACF (Partial AutoCorrelation Function) pour identifier ces paramètres."""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF et PACF pour choisir p et q
plot_acf(monthly_sales_diff)
plot_pacf(monthly_sales_diff)
plt.show()

"""Construction du modèle ARIMA
Une fois les paramètres choisis, vous pouvez ajuster le modèle ARIMA. Par exemple, supposons que les paramètres p=1, d=1, et q=1 sont adéquats :"""

from statsmodels.tsa.arima.model import ARIMA

# Ajuster le modèle ARIMA
model = ARIMA(monthly_sales, order=(1, 1, 1))
model_fit = model.fit()

# Résumé du modèle
print(model_fit.summary())

"""Prédiction des ventes pour les 6 prochains mois
Maintenant que le modèle est ajusté, vous pouvez effectuer des prévisions pour les 6 prochains mois :"""
# Prédiction des 6 prochains mois
forecast = model_fit.forecast(steps=6)
print("Prédictions des ventes pour les 6 prochains mois :")
print(forecast)

"""Visualisation des prévisions
Pour visualiser les prévisions, vous pouvez les tracer par rapport aux données historiques :"""
# Ajouter les prévisions aux données historiques
forecast_index = pd.date_range(monthly_sales.index[-1], periods=7, freq='M')[1:]
forecast_series = pd.Series(forecast, index=forecast_index)

# Tracer les prévisions
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales, label='Ventes historiques')
plt.plot(forecast_series, label='Prédictions de ventes', color='red')
plt.title('Prédiction des ventes pour les 6 prochains mois')
plt.xlabel('Date')
plt.ylabel('Montant des ventes')
plt.legend()
plt.show()

# Evaluation des performances prédictives
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import numpy as np

# # Supposons que 'test_data' contient les valeurs réelles de ventes pour les derniers mois
# mae = mean_absolute_error(test_data, forecast)
# rmse = np.sqrt(mean_squared_error(test_data, forecast))
# print(f"MAE: {mae}")
# print(f"RMSE: {rmse}")

