# IMPORTS
import pandas as pd  # MANEJO DE DATAFRAMES
from sklearn.impute import SimpleImputer  # IMPUTACIÓN DE VALORES NULOS
import json  # GUARDAR LISTAS DE COLUMNAS

# CARGAR DATASET ORIGINAL
df = pd.read_csv('../../data/dataset.csv')
print(f"[ INFO ] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# SEPARAR COLUMNAS NUMÉRICAS Y CATEGÓRICAS
num_cols = df.select_dtypes(include=['int64','float64']).columns
cat_cols = df.select_dtypes(include=['object','category']).columns
print(f"[ INFO ] Columnas numéricas: {len(num_cols)}, Columnas categóricas: {len(cat_cols)}")

# GUARDAR LISTA DE COLUMNAS ORIGINALES
columns_info = {
    'categorical': list(cat_cols),
    'numeric_median': list(num_cols),  # se imputarán con mediana
    'categorical_mode': list(cat_cols)  # se imputarán con moda
}

# IMPUTAR NUMÉRICAS CON MEDIANA
imputer_num = SimpleImputer(strategy='median')
df[num_cols] = imputer_num.fit_transform(df[num_cols])
print("[ INFO ] Valores nulos numéricos imputados con mediana")

# IMPUTAR CATEGÓRICAS CON MODA
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
print("[ INFO ] Valores nulos categóricos imputados con moda")

# GUARDAR DATASET TRATADO
output_csv = '../../results/preparation/02_nulls.csv'
df.to_csv(output_csv, index=False)
print(f"[ GUARDADO ] Dataset sin nulos en '{output_csv}'")

# INFORMACIÓN FINAL DEL DATASET YA SIN NULOS
print(f"[ INFO ] Dataset final sin nulos: {df.shape[0]} filas, {df.shape[1]} columnas")

# GUARDAR LISTA DE COLUMNAS EN JSON
output_json = '../../results/preparation/02_aux.json'
with open(output_json, 'w') as f:
    json.dump(columns_info, f, indent=4)
print(f"[ GUARDADO ] Columnas categóricas y numéricas imputadas en '{output_json}'")
