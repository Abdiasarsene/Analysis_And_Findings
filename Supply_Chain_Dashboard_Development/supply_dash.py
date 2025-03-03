# Importing librairies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Importing dataset
supply = pd.read_excel(r"D:\Projects\IT\Data Science & IA\datascience\supply_chain_data.xlsx")

# Print database
supply.head()
supply.info()

# Print data missing
dataMissing_supply = msno.bar(supply, color='cyan')

# Deletion of categoricals variables into dataset
suppli = supply.select_dtypes(exclude=['object']).drop(columns=['Delivery Date', 'Estimated Delivery Date'])

suppli

# Outliers detection by IQR method
number_data = suppli.select_dtypes(include={'number'})
Q1= number_data.quantile(0.25)
Q3 = number_data.quantile(0.75)
IQR = Q3 - Q1
outliers_detection = (number_data<(Q1-(1.5*IQR))|number_data>(Q3+(1.5*IQR)))
outliers = suppli[outliers_detection(any=1)]
outliers