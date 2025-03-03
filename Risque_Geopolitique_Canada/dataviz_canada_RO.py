# IMPORTATION DES DONNEES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# IMPORTATION DE LA BASE DE DONNEES
canada_RO = pd.read_excel(r"c:\Users\HP ELITEBOOK 840 G6\Downloads\canada_data_cleaned.xlsx")

# SUPPRESSION DES VARIABLES INUTILES
variables = ['Log_actifs_t', 'ROA_t-1_emprunteur','Valeur_marché_t-1', 'Nombre_chefs_de_file', 'Log_nombre_prêteurs','Log_taille_t-1', 'Ratio_capital_t-1', 'ROA_t-1_prêteur',
'Total_dépôts_actifs_t-1','Total_prêts_actifs_t-1', 'Années']
canada_RO = canada_RO.drop(columns=variables)

# PREPARATION DES DONNEES
check_missing = canada_RO.isna().sum() # Vérification des valeurs manquantes
check_dupli = canada_RO.duplicated().sum() #Vérification des doublons

# SUPPRESSION DES DOUBLONS
dupli = canada_RO[canada_RO.duplicated()]
if dupli.empty :
    print('Base de données nickel')
else :
    print("Des doublons ont été localisés dans la base de données. Passons à la supressions de ces derniers ")
    canada_RO_cleaned = canada_RO.drop_duplicates()

# DETECTION DES OUTLIERS
Q1 = canada_RO_cleaned.quantile(0.25)
Q3 = canada_RO_cleaned.quantile(0.75)
IQR = Q3 - Q1
outlier = ((canada_RO_cleaned< (Q1 - 1.5 * IQR)) | (canada_RO_cleaned > (Q3 + 1.5 * IQR)))
valeurs_aberrantes = canada_RO_cleaned[outlier.any(axis=1)] 
canada_RO_clean=canada_RO_cleaned.drop(valeurs_aberrantes.index)
canada_RO_clean

# NORMALISATION DES DONNEES
variable = ['Proportion_prêts_retenue', 'Maturité_prêt', 'Risque_géopolitique','Log_montant_transaction', 'Garantie', 'Fonds_de_roulement','Acquisition', 'Refinancement']
scaler = MinMaxScaler()
canada_RO_norma =canada_RO_clean[variable]
canada_RO_norma[variable]= scaler.fit_transform(canada_RO_clean[variable])
canada_RO_norma

# ANALYSE EXPLORATOIRE DES DONNEES
statistic = canada_RO_norma.describe() #Statistique descriptive
correlation = canada_RO_norma.corr() #corrélation
sns.heatmap(correlation, annot=True, cmap='magma')#matrice thermique de la corrélation
sns.pairplot(canada_RO_norma)#visualisation des variables

# Visualisation des relations entre la cible et les prédicteurs
# Liste des prédicteurs
predicteurs = ['Maturité_prêt', 'Risque_géopolitique', 'Log_montant_transaction','Acquisition']

# Création de la figure et des axes
fig, axes = plt.subplots(2,1, figsize=(40, 25))
axes = axes.flatten()

# Couleurs variées
colors = sns.color_palette("husl", len(variables))

# Boucle pour créer les nuages de points
for i, var in enumerate(predicteurs):
    sns.regplot(x=var, y='Proportion_prêts_retenue', data=canada_RO_norma, ax=axes[i], color=colors[i])
    axes[i].set_title(f'{var} vs Proportion_prêts_retenue')

# Supprimer le dernier axe vide si nécessaire
fig.delaxes(axes[-1])

# Les autres prédicteurs
fig, ax = plt.subplots(1,2, figsize =(14,6))
sns.regplot(data=canada_RO_norma, x='Fonds_de_roulement', y= 'Proportion_prêts_retenue', color='purple', ax=ax[0] )
ax[0].set(
    title=('Relation entre la proportion des prêts retenus et le fonds de roulement'),
    xlabel=('Fonds_de_roulement'),
    ylabel=('Proportion_prêts_retenue')
)
sns.regplot(data=canada_RO_norma, x='Refinancement', y= 'Proportion_prêts_retenue', color='orange', ax=ax[1] )
ax[1].set(
    title=('Relation en les revenus nets et le refinancement'),
    xlabel=('Refinancement'),
    ylabel=('Proportion_prêts_retenue')
)
sns.regplot(data=canada_RO_norma, x='Garantie', y= 'Proportion_prêts_retenue', color='black' )
# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()
plt.show()