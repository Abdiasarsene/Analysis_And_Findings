# Importation des librairies
import pandas as pd
import missingno as msno
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from googletrans import Translator
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Importation de la base de données
article = pd.read_excel(r'D:\Projets\Projet IT\Projet Informatique\datascience\projects\scientific - 20461\data\scientific_data.xlsx')
article

# Setting the aesthetic style of the plots
sns.set_style("whitegrid")
plt.figure(figsize=(14, 7))

# Line plot for 'Purchasing Power'
sns.lineplot(x='Years', y='Geopolitical risk', article= article, label='Geopolitical risk')

# Line plot for 'Food Prices'
sns.lineplot(x='Years', y='Food prices index', article= article, label='Food Prices')

# Adding labels and title
plt.xlabel('Years')
plt.ylabel('Value')
plt.title('Evolution of Geopolitical risk and Food Prices')
plt.legend()

# Display the plot
plt.show()
"Purchasing power gross disposable income"

# Setting the aesthetic style of the plots
sns.set_style("darkgrid")
plt.figure(figsize=(14, 7))

# Line plot for 'Geopolitical Risk'
sns.lineplot(x='Years', y='Geopolitical risk', article=article, label='Geopolitical Risk')

# Line plot for 'Economic Inequality (Gini coefficient)'
sns.lineplot(x='Years', y='Economic inequality (Gini coefficient)', article=article, label='Economic Inequality (Gini Coefficient)')

# Line plot for 'Unemployment Rate' on secondary y-axis
ax = sns.lineplot(x='Years', y='Unemployment rate', article=article, label='Unemployment Rate', color='red')
ax.set_ylabel('Unemployment Rate (%)')
ax.legend(loc='upper left')

# Adding labels and title
plt.xlabel('Years')
plt.ylabel('Value')
plt.title('Impact of Geopolitical Tensions and Economic Inequality on Unemployment Rate')
plt.legend()

# Display the plot
plt.show()


# Préparation des données (supposons que les données sont déjà chargées dans article)
x = article[['Average temperature', 'Geopolitical risk', 'Carbon tax', 'Economic inequality (Gini coefficient)', 'Annual GDP growth rate', 'Unemployment rate', 'Inflation rate', 'Purchasing power gross disposable income']]  # Prédicteurs
y = article['Food prices index']  # Variable cible
date = article['Years']  # Date

# Ajuster le modèle de régression linéaire
model = sm.OLS(y, x).fit()

# Extraction des coefficients et p-values
coefficients = model.params
p_values = model.pvalues

# Déterminer les significativités
significance_level = 0.05
significant = p_values < significance_level

# Créer un DataFrame pour les résultats
results = pd.DataFrame({
    'Coefficients': coefficients,
    'P-values': p_values,
    'Significant': significant
})

# Configuration de Seaborn
sns.set(style="whitegrid")

# Création de la figure
plt.figure(figsize=(10, 6))
palette = {True: "green", False: "red"}

# Tracé des coefficients
sns.barplot(x=results.index, y=results['Coefficients'], hue=results['Significant'], dodge=False, palette=palette)

# Ajouts esthétiques
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('Predictors')
plt.ylabel('Coefficients')
plt.title('Estimation of the regression model with structural break')
plt.legend(title='Significant', loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Afficher la figure
plt.show()

# Visualisation graphique
y = article['Food prices index']  # Variable cible
date = article['Years']  # Date

# Estimation d'un modèle de régression linéaire avec une rupture structurelle
breakpoint = 50  # On suppose une rupture à t = 50
article['intercept'] = 1
article['slope_before'] = np.where(date < breakpoint, date, 0)
article['slope_after'] = np.where(date >= breakpoint, date - breakpoint, 0)
model = sm.OLS(y, x).fit()

# Création du DataFrame pour Seaborn
data = pd.DataFrame({
    'Years': date,
    'Observed values': y,
    'Adjusted values': model.fittedvalues
})

# Configuration de Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Graphique de Seaborn
sns.lineplot(x='Years', y='Observed values', data=data, label='Observed values', marker='o')
sns.lineplot(x='Years', y='Adjusted values', data=data, label='Adjusted values', color='red', marker='x')

# Ligne pour la rupture structurelle
plt.axvline(date.iloc[breakpoint], color='green', linestyle='--', label='Rupture structurelle')

# Ajouts esthétiques
plt.xlabel('Years')
plt.ylabel('Food prices index')
plt.legend()
plt.title('Analysis of structural breaks')
plt.tight_layout()

plt.show()

# Supposons que 'model' est déjà ajusté à partir des données précédentes
# Calcul des résidus
residuals = model.resid
fitted_values = model.fittedvalues

# Création du DataFrame pour Seaborn
residuals_data = pd.DataFrame({
    'Adjusted value': fitted_values,
    'Residuals': residuals
})

# Configuration de Seaborn
sns.set(style="whitegrid")

# Création de la figure
plt.figure(figsize=(12, 6))

# Graphique de Seaborn pour les résidus
sns.residplot(x='Adjusted value', y='Residuals', data=residuals_data, lowess=True, 
              line_kws={'color': 'red', 'lw': 1}, scatter_kws={'alpha': 0.5})

# Ajouts esthétiques
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('Adjusted values')
plt.ylabel('Residuals')
plt.title('Residuals analysis')
plt.tight_layout()

# Afficher la figure
plt.show()

# Supposons que les 'residuals' soient déjà calculés
# Configuration de Seaborn
sns.set(style="whitegrid")

# Création de la figure Q-Q plot
plt.figure(figsize=(10, 6))
sm.qqplot(residuals, line='s', ax=plt.gca(), markerfacecolor='blue', markeredgecolor='blue', alpha=0.7)

# Ajouts esthétiques
plt.title('Q-Q Plot des résidus', fontsize=15, weight='bold')
plt.xlabel('Theoretical quantiles', fontsize=12)
plt.ylabel('Residuals quantiles', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

# Affichage de la figure
plt.show()



# Prédiction des prix alimentaires
# Séparation des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialisation et ajustement du modèle de régression linéaire
model = LinearRegression()
model.fit(x_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(x_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Visualisation des résultats
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Real values ​​of food prices')
plt.ylabel('Predicted values ​​of food prices')
plt.title('Food Price Prediction')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualisation des coefficients du modèle
coefficients = model.coef_
features = x.columns
coef_df = pd.DataFrame({
    'Features': features,
    'Coefficients': coefficients
}).sort_values(by='Coefficients')

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficients', y='Features', data=coef_df, palette='viridis')
plt.xlabel('Regression coefficient')
plt.ylabel('Features')
plt.title('Importance of characteristics for food price prediction')
plt.tight_layout()
plt.show()

# Réseaux de neureones
# Initialisation et ajustement du réseau de neurones
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(x_train, y_train)

# Prédiction
y_pred_mlp = mlp.predict(x_test)

# Évaluation
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)
print(f'MLP - Mean Squared Error: {mse_mlp}')
print(f'MLP - R^2 Score: {r2_mlp}')
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x=y_test, y=y_pred_mlp, alpha=0.5, color='green', label='MLP') 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', linewidth=2) 
plt.xlabel('Real values ​​of food prices') 
plt.ylabel('Predicted values ​​of food prices') 
plt.title('Food Price Prediction with Neural Network (MLP)') 
plt.legend() 
plt.show()


# Forêt aléatoire
# Initialisation et ajustement du modèle de forêts aléatoires
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

# Prédiction
y_pred_rf = rf.predict(x_test)

# Évaluation
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest - Mean Squared Error: {mse_rf}')
print(f'Random Forest - R^2 Score: {r2_rf}')

# Visualisation graphique
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.5, color='orange', label='Random Forests') 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', linewidth=2) 
plt.xlabel('Real values ​​of food prices') 
plt.ylabel('Predicted values ​​of food prices') 
plt.title('Food Price Prediction with Random Forests') 
plt.legend() 
plt.tight_layout() 
plt.show()

# Optimisations des hyperparamètres
# Modèle de régression linéaire
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_test)

# Optimisation des hyperparamètres pour MLP
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'max_iter': [200, 500]
}
grid_mlp = GridSearchCV(MLPRegressor(random_state=42), param_grid_mlp, cv=5, scoring='r2')
grid_mlp.fit(x_train, y_train)
y_pred_mlp_opt = grid_mlp.predict(x_test)

# Optimisation des hyperparamètres pour les forêts aléatoires
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt']
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='r2')
grid_rf.fit(x_train, y_train)
y_pred_rf_opt = grid_rf.predict(x_test)

# Configuration de Seaborn
sns.set(style="whitegrid")

# Création des figures séparées pour chaque modèle

# Régression linéaire
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_linear, alpha=0.5, color='red', label='Linear regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', linewidth=2)
plt.xlabel('Real values ​​of food prices')
plt.ylabel('Predicted values ​​of food prices')
plt.title('Food Price Prediction with Linear Regression')
plt.legend()
plt.tight_layout()
plt.show()

# MLP optimisé
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_mlp_opt, alpha=0.5, color='purple', label='MLP optimized')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', linewidth=2)
plt.xlabel('Real values ​​of food prices')
plt.ylabel('Predicted values ​​of food prices')
plt.title('Food Price Prediction with Optimized MLP')
plt.legend()
plt.tight_layout()
plt.show()

# Forêts Aléatoires optimisées
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf_opt, alpha=0.5, color='brown', label='Optimized Random Forests')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', linewidth=2)
plt.xlabel('Real values ​​of food prices')
plt.ylabel('Predicted values ​​of food prices')
plt.title('Food Price Prediction with Optimized Random Forests')
plt.legend()
plt.tight_layout()
plt.show()

# Visualisation de tous les modèles de prédiction

# Création du DataFrame pour les résultats
results = pd.DataFrame({
    'Actual Values': y_test,
    'Linear Model': y_pred_linear,
    'MLP': y_pred_mlp,
    'Random Forest': y_pred_rf,
    'Optimized MLP': y_pred_mlp_opt,
    'Optimized Random Forest': y_pred_rf_opt
})

# Configuration de Seaborn
sns.set(style="whitegrid")

# Création de la figure
plt.figure(figsize=(14, 8))

# Tracé des valeurs prédites par les différents modèles
sns.scatterplot(x='Actual Values', y='Actual Values', data=results, color='blue', label='Real values')
sns.scatterplot(x='Actual Values', y='Linear Model', data=results, color='red', label='Linear Regression')
sns.scatterplot(x='Actual Values', y='MLP', data=results, color='green', label='MLP')
sns.scatterplot(x='Actual Values', y='Random Forest', data=results, color='orange', label='Random Forest')
sns.scatterplot(x='Actual Values', y='Optimized MLP', data=results, color='purple', label='MLP Optimized')
sns.scatterplot(x='Actual Values', y='Optimized Random Forest', data=results, color='brown', label='Optimized Random Forest')

# Ajouts esthétiques
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', linewidth=2)
plt.xlabel('Real values ​​of food prices')
plt.ylabel('Predicted values ​​of food prices')
plt.title('Comparison of food price prediction models')
plt.legend()
plt.tight_layout()

# Affichage de la figure
plt.show()
