# src/02_tratamiento_nulos.py
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('../../data/dataset.csv')

# Separar numéricas y categóricas
num_cols = df.select_dtypes(include=['int64','float64']).columns
cat_cols = df.select_dtypes(include=['object','category']).columns

# Imputar numéricas con mediana
imputer_num = SimpleImputer(strategy='median')
df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Imputar categóricas con moda
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

df.to_csv('../../results/preparation/02_tratamiento_nulos.csv', index=False)
print("Dataset sin nulos guardado en '../../results/preparation/02_tratamiento_nulos.csv'")
