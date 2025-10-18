import pandas as pd
import numpy as np
import os

# PARÁMETROS CONFIGURABLES
INPUT_CSV = '../../results/preparation/05_variance.csv'       # DATASET DE ENTRADA
OUTPUT_CSV = '../../results/preparation/00_contaminated.csv'  # DATASET DE SALIDA
RESULTS_FOLDER = '../../results/preparation'                  # CARPETA DE RESULTADOS

CONTAMINATION_RATE = 0.02      # PORCENTAJE DE FILAS A CONTAMINAR
NOISE_INTENSITY = 3.0          # FACTOR DE RUIDO (DESVIACIÓN)
ALL_COLUMNS = True             # TRUE = TODAS LAS COLUMNAS NUMÉRICAS, FALSE = SOLO LAS ESPECIFICADAS
COLUMNS_TO_CONTAMINATE = ['pressure', 'flow_rate']  # COLUMNAS A CONTAMINAR SI ALL_COLUMNS=FALSE
CONTAMINATION_SCOPE = True      # TRUE = CONTAMINAR TODAS LAS COLUMNAS SELECCIONADAS, LISTA = COLUMNAS ESPECÍFICAS
ADD_LABEL = True               # TRUE = AÑADIR COLUMNA is_anomaly
RANDOM_STATE = 42              # SEMILLA PARA REPRODUCIBILIDAD
SHOW_INFO = True               # TRUE = MOSTRAR INFO EN CONSOLA

# CREAR CARPETA DE RESULTADOS
os.makedirs(RESULTS_FOLDER, exist_ok=True)
if SHOW_INFO:
    print(f"[ INFO ] Carpeta '{RESULTS_FOLDER}' creada si no existia")

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# SELECCIONAR COLUMNAS A CONTAMINAR
if ALL_COLUMNS:
    cols_to_contaminate = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
else:
    cols_to_contaminate = [c for c in COLUMNS_TO_CONTAMINATE if c in df.columns]

# SI CONTAMINATION_SCOPE ES LISTA, FILTRAR COLUMNAS
if isinstance(CONTAMINATION_SCOPE, list):
    cols_to_contaminate = [c for c in CONTAMINATION_SCOPE if c in df.columns]

# DETECTAR COLUMNAS NO EXISTENTES
ignored_cols = [c for c in (COLUMNS_TO_CONTAMINATE if not ALL_COLUMNS else []) if c not in df.columns]

if not cols_to_contaminate:
    raise ValueError("[ ERROR ] NO SE ENCONTRARON COLUMNAS VÁLIDAS PARA CONTAMINAR")

# MOSTRAR INFO
if SHOW_INFO:
    print(f"[ INFO ] Numero de columnas a contaminar: {len(cols_to_contaminate)}")
    if ignored_cols:
        print(f"[ INFO ] Columnas ignoradas (no existe en el dataset): {ignored_cols}")

# DETERMINAR FILAS A CONTAMINAR
np.random.seed(RANDOM_STATE)
n_rows = df.shape[0]
n_contam = int(n_rows * CONTAMINATION_RATE)
contam_indices = np.random.choice(df.index, size=n_contam, replace=False)

if SHOW_INFO:
    print(f"[ INFO ] Filas a contaminar: {n_contam}")

# COPIAR DATAFRAME ORIGINAL
df_contaminated = df.copy()

# APLICAR CONTAMINACIÓN
for col in cols_to_contaminate:
    if df[col].dtype in ['int64', 'float64']:
        noise = np.random.normal(0, NOISE_INTENSITY * df[col].std(), size=n_contam)
        df_contaminated.loc[contam_indices, col] += noise

if SHOW_INFO:
    print(f"[ info ] Contaminación aplicada en columnas seleccionadas")

# AÑADIR ETIQUETA DE ANOMALÍA
if ADD_LABEL:
    df_contaminated['is_anomaly'] = 0
    df_contaminated.loc[contam_indices, 'is_anomaly'] = 1
    if SHOW_INFO:
        print("[ info ] Columna 'is_anomaly' añadida (1=anomalía)")

# GUARDAR RESULTADOS
df_contaminated.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ guardado ] Dataset contaminado en '{OUTPUT_CSV}'")
