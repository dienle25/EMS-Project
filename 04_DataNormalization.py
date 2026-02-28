import pandas as pd

df=pd.read_csv('04_Normalized_Data.csv')

#print(df.head())
#df.info()
#print(df.describe(include='all'))

print(df.drop_duplicates())