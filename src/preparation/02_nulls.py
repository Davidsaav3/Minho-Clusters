import pandas as pd  # PARA MANEJO DE DATAFRAMES
from sklearn.impute import SimpleImputer  # PARA IMPUTAR VALORES NULOS
import json  # PARA GUARDAR INFORMACIÓN DE COLUMNAS EN FORMATO JSON

# PARÁMETROS CONFIGURABLES
INPUT_FILE = '../../data/dataset.csv'                  # RUTA DEL DATASET ORIGINAL
OUTPUT_CSV = '../../results/preparation/02_nulls.csv'  # RUTA DEL DATASET FINAL SIN NULOS
OUTPUT_JSON = '../../results/preparation/02_aux.json'  # RUTA DEL JSON CON INFORMACIÓN DE COLUMNAS
NUMERIC_STRATEGY = 'median'                            # ESTRATEGIA DE IMPUTACIÓN PARA COLUMNAS NUMÉRICAS ('mean','median','constant')
CATEGORICAL_STRATEGY = 'most_frequent'                 # ESTRATEGIA DE IMPUTACIÓN PARA COLUMNAS CATEGÓRICAS ('most_frequent','constant')
FILL_CONSTANT_NUMERIC = 0                               # VALOR A USAR SI strategy='constant' PARA NUMÉRICOS
FILL_CONSTANT_CATEGORICAL = 'unknown'                  # VALOR A USAR SI strategy='constant' PARA CATEGÓRICOS
REMOVE_EMPTY_COLUMNS = True                             # ELIMINAR COLUMNAS COMPLETAMENTE VACÍAS
SHOW_INFO = True                                       # MOSTRAR MENSAJES INFORMATIVOS EN PANTALLA
SAVE_INTERMEDIATE = False                              # GUARDAR CSV INTERMEDIO ANTES DE IMPUTAR NULOS

# CARGAR DATASET
df = pd.read_csv(INPUT_FILE)  # LEER EL CSV DE ENTRADA
if SHOW_INFO:
    print(f"[ INFO ] Dataset cargado: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# IDENTIFICAR COLUMNAS POR TIPO
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()  # DETECTAR COLUMNAS NUMÉRICAS
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()  # DETECTAR COLUMNAS CATEGÓRICAS
if SHOW_INFO:
    print(f"[ INFO ] Columnas numéricas: {len(num_cols)}, Columnas categóricas: {len(cat_cols)}")

# GUARDAR INFORMACIÓN INICIAL DE COLUMNAS
columns_info = {
    'categorical': cat_cols,                # LISTA DE COLUMNAS CATEGÓRICAS DETECTADAS
    'numeric_imputed': num_cols,            # COLUMNAS NUMÉRICAS QUE SE IMPUTARÁN
    'categorical_imputed': cat_cols,        # COLUMNAS CATEGÓRICAS QUE SE IMPUTARÁN
    'removed_empty_columns': []             # COLUMNAS VACÍAS QUE SE ELIMINARÁN
}

# ELIMINAR COLUMNAS COMPLETAMENTE VACÍAS (OPCIONAL)
if REMOVE_EMPTY_COLUMNS:
    empty_cols = df.columns[df.isna().all()].tolist()  # DETECTAR COLUMNAS VACÍAS
    df.drop(columns=empty_cols, inplace=True)          # ELIMINAR COLUMNAS VACÍAS
    columns_info['removed_empty_columns'] = empty_cols  # GUARDAR LISTA DE COLUMNAS ELIMINADAS
    if SHOW_INFO:
        print(f"[ INFO ] Columnas completamente vacías eliminadas: {len(empty_cols)}")

# GUARDAR DATASET INTERMEDIO (OPCIONAL)
if SAVE_INTERMEDIATE:
    intermediate_csv = OUTPUT_CSV.replace('.csv','_intermediate.csv')  # NOMBRE DEL CSV INTERMEDIO
    df.to_csv(intermediate_csv, index=False)  # GUARDAR DATASET INTERMEDIO
    if SHOW_INFO:
        print(f"[ GUARDADO ] Dataset intermedio en '{intermediate_csv}'")

# IMPUTAR COLUMNAS NUMÉRICAS
num_strategy_params = {'strategy': NUMERIC_STRATEGY}  # CONFIGURAR ESTRATEGIA
if NUMERIC_STRATEGY == 'constant':  # SI USAMOS CONSTANTE, DEFINIR VALOR
    num_strategy_params['fill_value'] = FILL_CONSTANT_NUMERIC

imputer_num = SimpleImputer(**num_strategy_params)  # CREAR OBJETO IMPUTER
df[num_cols] = imputer_num.fit_transform(df[num_cols])  # IMPUTAR NULOS NUMÉRICOS
if SHOW_INFO:
    print(f"[ INFO ] Valores nulos numéricos imputados con '{NUMERIC_STRATEGY}'")

# IMPUTAR COLUMNAS CATEGÓRICAS
cat_strategy_params = {'strategy': CATEGORICAL_STRATEGY}  # CONFIGURAR ESTRATEGIA
if CATEGORICAL_STRATEGY == 'constant':  # SI USAMOS CONSTANTE, DEFINIR VALOR
    cat_strategy_params['fill_value'] = FILL_CONSTANT_CATEGORICAL

imputer_cat = SimpleImputer(**cat_strategy_params)  # CREAR OBJETO IMPUTER
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])  # IMPUTAR NULOS CATEGÓRICOS
if SHOW_INFO:
    print(f"[ INFO ] Valores nulos categóricos imputados con '{CATEGORICAL_STRATEGY}'")

# GUARDAR DATASET FINAL
df.to_csv(OUTPUT_CSV, index=False)  # GUARDAR CSV FINAL SIN NULOS
if SHOW_INFO:
    print(f"[ GUARDADO ] Dataset final sin nulos en '{OUTPUT_CSV}'")
    print(f"[ INFO ] Dataset final: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# GUARDAR INFORMACIÓN DE COLUMNAS EN JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(columns_info, f, indent=4)  # GUARDAR COLUMNAS IMPUTADAS Y ELIMINADAS
if SHOW_INFO:
    print(f"[ GUARDADO ] Columnas imputadas guardadas en '{OUTPUT_JSON}'")
