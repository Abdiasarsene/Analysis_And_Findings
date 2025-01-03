import pandas as pd

data = pd.read_excel(r"D:\Projets\Projet IT\Projet Datascience\reelproject\projects\abdiel\diel_data.xlsx")

data_rouded = round(data)
data_rouded

data_rouded.to_excel('datas.xlsx', index=False)