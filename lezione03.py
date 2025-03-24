import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv("HR.csv")
'''
print(df.head())


print(df.info())

print(df.describe())
''''''
# Controlla i valori nulli
print("Valori nulli per colonna:")
print(df.isnull().sum())

# Rimuovi duplicati
print("\nNumero di righe duplicate:", df.duplicated().sum())
df = df.drop_duplicates()



#boxplots
# Seleziona solo colonne numeriche
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Crea i boxplots
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols  # Divisione al sovrappiù
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
axes = axes.ravel()  # Aplanare l'array 2D in un 1D

for i, col in enumerate(numeric_cols):
    df.boxplot(column=col, vert=True, patch_artist=True, ax=axes[i])
    axes[i].set_title(f'Colonna {col}')
    axes[i].grid(True)

# Rimuovi subplot vuoti
for j in range(len(numeric_cols), n_rows * n_cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Se sono presenti valori nulli, scegli un metodo per gestirli (es. riempimento con media/mediana)
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)  # Usa mediana o media
        
# Controlla di nuovo dopo la pulizia
print("\nDuplicati rimanenti:", df.duplicated().sum())

from sklearn.preprocessing import LabelEncoder

# Esempio: Codifica colonna 1 (Attrition: "Yes"/"No")
le_attrition = LabelEncoder()
df[1] = le_attrition.fit_transform(df[1])

# Esempio: Codifica colonna 2 (BusinessTravel: "Travel_Rarely", "Travel_Frequently", "Non-Travel")
df = pd.get_dummies(df, columns=[2], drop_first=True)

# Ripeti per altre colonne categoriche (es. colonne 4, 7, 12, 14, 18)

import seaborn as sns

# Calcola correlazione tra colonne numeriche
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Matrice di Correlazione")
plt.show()

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribuzione di Colonna {col}")
    plt.show()



# Conta i valori di Attrition
attrition_counts = df['Attrition'].value_counts()
attrition_percent = df['Attrition'].value_counts(normalize=True) * 100

print("\nConteggio dei valori:\n", attrition_counts)
print("\nPercentuali:\n", attrition_percent)

# Grafico a barre
plt.figure(figsize=(8, 5))
attrition_counts.plot(kind='bar', color=['salmon', 'lightgreen'])
plt.title('Distribuzione della variabile Target "Attrition"')
plt.xlabel('Valore (Yes/No)')
plt.ylabel('Numero di dipendenti')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#commento: La colonna Attrition del dataset è sbilanciato sul "NO".
'''

'''
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE  # Per bilanciare i dati

# 1. Carica il dataset
df = pd.read_csv('HR.csv')

# 2. Rimuovi colonne costanti (es. EmployeeCount sempre 1)
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=constant_cols)

# 3. Separazione in features e target
y = df['Attrition']
X = df.drop('Attrition', axis=1)

# 4. Codifica la variabile target (Attrition)
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 0 = "No", 1 = "Yes"

# 5. Identifica colonne categoriche e numeriche
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# 6. Preprocessing con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'  # Mantiene le colonne numeriche
)

# 7. Applica il preprocessing
X_processed = preprocessor.fit_transform(X)

# 8. Bilanciamento del dataset con SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Bilancia la classe "Yes" al 50%
X_resampled, y_resampled = smote.fit_resample(X_processed, y_encoded)

# 9. Divisione in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# 10. Calcola scale_pos_weight per XGBoost
num_negative = np.sum(y_resampled == 0)
num_positive = np.sum(y_resampled == 1)
scale_pos = num_negative / num_positive

# 11. Addestra il modello XGBoost con bilanciamento
model = XGBClassifier(
    scale_pos_weight=scale_pos,  # Peso per la classe "Yes"
    max_depth=4,                # Profondità degli alberi
    learning_rate=0.1,          # Passo di apprendimento
    n_estimators=200,           # Numero di alberi
    use_label_encoder=False,    # Evita warning
    eval_metric='logloss'       # Metrica di valutazione
)
model.fit(X_train, y_train)

# 12. Genera predizioni e probabilità
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 13. Valutazione del modello
# a) Report di Classificazione
print("Report di Classificazione:\n")
print(classification_report(y_test, y_pred))

# b) Matrice di Confusione
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione')
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.show()

# c) Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

# d) Feature Importances
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    # Ottieni i nomi delle colonne dopo il preprocessing
    feature_names = (
        numerical_cols + 
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
    )
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Importanza delle Features")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Salvataggio del modello e del preprocessore
import pickle

# Salva il modello
with open('xgboost_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Salva il preprocessore
with open('preprocessor.pkl', 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)  # Save the LabelEncoder instance
'''

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Carica il preprocessore, il modello e il LabelEncoder
with open('preprocessor.pkl', 'rb') as preprocessor_file:
    loaded_preprocessor = pickle.load(preprocessor_file)

with open('xgboost_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    loaded_le = pickle.load(le_file)

# Carica il nuovo dataset (deve avere la colonna 'Attrition')
new_data = pd.read_csv('HR.csv')  # Assicurati che il file esista

# Separazione in features e target
X_new = new_data.drop('Attrition', axis=1)
y_new = new_data['Attrition']

# Preprocessing delle features
X_new_processed = loaded_preprocessor.transform(X_new)

# Predizioni e probabilità
y_pred_encoded = loaded_model.predict(X_new_processed)
y_proba = loaded_model.predict_proba(X_new_processed)[:, 1]  # Probabilità per la classe "Yes"

# Codifica della target reale (Attrition)
y_true_encoded = loaded_le.transform(y_new)

# --- Valutazione ---
# 1. Report di Classificazione
print("Report di Classificazione:")
print(classification_report(y_true_encoded, y_pred_encoded, 
                            target_names=['No', 'Yes']))

# 2. Matrice di Confusione
cm = confusion_matrix(y_true_encoded, y_pred_encoded)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione')
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.show()

# 3. Curva ROC
fpr, tpr, thresholds = roc_curve(y_true_encoded, y_proba)
roc_auc = roc_auc_score(y_true_encoded, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC')
plt.xlabel('Tasso Falso Positivo (FPR)')
plt.ylabel('Tasso Veri Positivi (TPR)')
plt.legend()
plt.show()