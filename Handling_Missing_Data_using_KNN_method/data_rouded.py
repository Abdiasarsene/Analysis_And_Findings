# Importing librairie
import pandas as pd

# Importing dataset
data = pd.read_excel(r"D:\Projets\Projet IT\Projet Datascience\reelproject\projects\abdiel\diel_data.xlsx")

# Converting an round
data_rouded = round(data)
data_rouded

# Export data
data_rouded.to_excel('datas.xlsx', index=False)