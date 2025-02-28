import pandas as pd
df= pd.read_csv("iris.csv")
def df_info(df):
    print("informazioni del dataframe")
    print(df.info())
    
    print("Descrizione del Dataframe")
    print(df.describe())
    
    print("Prime righe del Dataframe")
    print(df.head())



#es 5
species = df["Species"]
print(species.head())

#es 6
df_filtrato= df[df["SepalLengthCm"] > 5.0]

print(df_filtrato.head())

#es 7
df['sepal_area'] = df['SepalLengthCm'] * df['SepalWidthCm']
print(df[['SepalLengthCm', 'SepalWidthCm', 'sepal_area']].head())

#es 8
df_ordinato = df.sort_values(by='PetalLengthCm', ascending=True)
print(df_ordinato.head())

#es 9
grouped = df.groupby('Species').mean()

print(grouped)

#es 10
pivot = df.pivot_table(index='Species', values='SepalLengthCm', aggfunc='count')

pivot = pivot.rename(columns={'sepal_length': 'count'})

print(pivot)

#es 11
import numpy as np


# Simula dei valori mancanti nelle prime 3 righe della colonna 'petal_length'

df.loc[0:2, 'PetalLengthCm'] = np.nan

print(df.head(5))


# Calcola la media della colonna escludendo i NaN

media_petal = df['PetalLengthCm'].mean()


# Sostituisci i NaN con la media calcolata

df['petal_length'] = df['PetalLengthCm'].fillna(media_petal)

print(df.head(5))

#es 12
# Crea un DataFrame con informazioni extra

df_extra = pd.DataFrame({

    'Species': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],

    'Color': ['red', 'blue', 'green']

})
print(df.head())

# Esegui il merge

df = pd.merge(df, df_extra, on='Species', how='left')

print(df.head())

#es 13
def categoriza_area(area):

    if area < 15:

        return 'small'

    else:

        return 'large'


df['sepal_category'] = df['sepal_area'].apply(categoriza_area)

print(df[['sepal_area', 'sepal_category']].head())


#es 14
import matplotlib.pyplot as plt


df.plot.scatter(x='SepalLengthCm', y='SepalWidthCm', title='Scatter plot: Sepal Length vs Sepal Width')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.show()

#es 15
# Salva il DataFrame in un nuovo file CSV

df.to_csv('iris_modificato.csv', index=False)


# Carica il file appena salvato per controllare

df_verifica = pd.read_csv('iris_modificato.csv')

print(df_verifica.head())

