import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
df= pd.read_csv("telco_with_nan.csv")
'''
#es 1
print(df.head())

#es 2
print(df.info())

#es 3
print(df.describe())

#es 4
customerID = df["customerID"]
print(customerID.head())

#es 5

sorted_tenure= df.sort_values(by="tenure", ascending=True)
print(sorted_tenure.head())



#es 6
print(df['TotalCharges'].dtype)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
righe_mancanti = (df['TotalCharges'].isna()).sum()
print(f"Righe con valori mancanti in 'TotalCharges': {righe_mancanti}")

mediana_TotalCharges = df["TotalCharges"].median()
df['TotalCharges']= df['TotalCharges'].fillna(mediana_TotalCharges)
righe_mancanti = (df['TotalCharges'].isna()).sum()
print(f"Righe con valori mancanti in 'TotalCharges' dopo riepimento con mediana: {righe_mancanti}")


#es 7
df.set_index('customerID', inplace=True)
print(df.duplicated().sum())

#es 8
Q1 = df["MonthlyCharges"].quantile(0.25)
Q3 = df["MonthlyCharges"].quantile(0.75)
IQR = Q3 - Q1

print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(y=df["MonthlyCharges"])
plt.title("Boxplot di MonthlyCharges")
plt.show()

# Limiti
lower_bound = Q1 - 0.5 * IQR
upper_bound = Q3 + 0.5 * IQR

# Identifica gli outlier
outlier_condition = (df["MonthlyCharges"] < lower_bound) | (df["MonthlyCharges"] > upper_bound)
outliers = df[outlier_condition]

print(f"Numero di outlier con 0.5×IQR: {len(outliers)}")

#es 9
# Pulizia della colonna "gender"
df["gender"] = df["gender"].astype(str).str.strip().str.title()
gender_map = {"M": "Male", "F": "Female"}
df["gender"] = df["gender"].replace(gender_map)

# Sostituisci "Nan" con np.nan e gestisci i valori mancanti
df["gender"] = df["gender"].replace("Nan", np.nan)
df["gender"] = df["gender"].fillna("Unknown")

# Verifica
print("Valori unici dopo la pulizia:")
print(df["gender"].unique())

#es 10
# Pulizia della colonna "tenure"
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df["tenure"].fillna(df["tenure"].median(), inplace=True)

# Min-Max Scaling
scaler = MinMaxScaler()
df["tenure_normalized"] = scaler.fit_transform(df[["tenure"]])

# Verifica il risultato
print("Range dei valori normalizzati:")
print(f"Min: {df['tenure_normalized'].min()}, Max: {df['tenure_normalized'].max()}")
print("\nEsempio di dati normalizzati:")
print(df[["tenure", "tenure_normalized"]].sample(10))

#es 11
# Crea la colonna "ChurnFlag"
df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0)

# Verifica il risultato
print("\nEsempio di dati:")
print(df[["Churn", "ChurnFlag"]].sample(5))

#es 12
# Calcola la media ignorando i NaN
average_charges_per_contract = df.groupby("Contract")["MonthlyCharges"].mean().reset_index()

# Rinomina e formatta
average_charges_per_contract.rename(columns={"MonthlyCharges": "Average_MonthlyCharges"}, inplace=True)
average_charges_per_contract["Average_MonthlyCharges"] = average_charges_per_contract["Average_MonthlyCharges"].round(2)

print(average_charges_per_contract)

#es 13

#pivot = df.pivot_table(index='InternetService', columns='Contract', aggfunc='sum')
#print(pivot)


# Creazione della pivot table con il conteggio dei clienti
pivot_table = df.pivot_table(
    index="InternetService",      # Righe: tipi di servizio internet
    columns="Contract",          # Colonne: tipi di contratto
    values="customerID",         # Valore da conteggiare (es. ID univoco dei clienti)
    aggfunc="count",             # Funzione di aggregazione: conteggio
    fill_value=0                 # Sostituisci i valori mancanti con 0
)

print(pivot_table)


#es 14
# DataFrame extra con mappatura dei contratti
df_extra = pd.DataFrame({
    'Contratto': ['Month-to-month', 'One year', 'Two year'],
    'Codice': [101, 102, 103]
})
print(df_extra)
# Unione corretta con left join
df_merged = pd.merge(
    left=df,
    right=df_extra,
    left_on='Contract',   # Colonna del DataFrame principale
    right_on='Contratto', # Colonna del DataFrame extra
    how='left'            # left join per mantenere tutte le righe di df
)

# Elimina la colonna "Contratto" duplicata (opzionale)
df_merged.drop(columns=['Contratto'], inplace=True)

print(df_merged.head())


#es 15
df = df.drop(columns=['customerID']) 
print(df.head())


#es 16
df['TotalCharges']= pd.to_numeric(df["TotalCharges"], errors="coerce")
df['TotalCharges'] = df['TotalCharges'].astype(float)
print(df['TotalCharges'].dtype)
'''