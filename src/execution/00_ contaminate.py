import pandas as pd
import numpy as np
import os

# PARÁMETROS CONFIGURABLES
INPUT_CSV = '../../results/preparation/05_variance.csv'       # ARCHIVO DE ENTRADA
OUTPUT_CSV = '../../results/execution/00_contaminated.csv'   # ARCHIVO DE SALIDA
RESULTS_FOLDER = '../../results/preparation'                  # CARPETA DE RESULTADOS
CONTAMINATION_RATE = 0.02      # PORCENTAJE DE FILAS A CONTAMINAR
NOISE_INTENSITY = 3.0          # INTENSIDAD DEL RUIDO
ALL_COLUMNS = True             # TRUE = TODAS COLUMNAS NUMÉRICAS
COLUMNS_TO_CONTAMINATE = ['pressure', 'flow_rate']  # COLUMNAS A CONTAMINAR SI ALL_COLUMNS=FALSE
CONTAMINATION_SCOPE = True      # TRUE = TODAS COLUMNAS SELECCIONADAS, LISTA = COLUMNAS ESPECÍFICAS
ADD_LABEL = True               # AÑADIR COLUMNA 'IS_ANOMALY'
RANDOM_STATE = 42              # SEMILLA PARA REPRODUCIBILIDAD
SHOW_INFO = True               # MOSTRAR INFORMACIÓN EN CONSOLA

# CREAR CARPETA DE RESULTADOS
os.makedirs(RESULTS_FOLDER, exist_ok=True)
if SHOW_INFO:
    print(f"[ INFO ] CARPETA '{RESULTS_FOLDER}' CREADA SI NO EXISTIA")

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# SELECCIONAR COLUMNAS A CONTAMINAR
if ALL_COLUMNS:
    cols_to_contaminate = df.select_dtypes(include=['int64', 'float64']).columns.tolist()  # TODAS COLUMNAS NUMÉRICAS
else:
    cols_to_contaminate = [c for c in COLUMNS_TO_CONTAMINATE if c in df.columns]  # FILTRAR COLUMNAS EXISTENTES

# FILTRAR COLUMNAS SI CONTAMINATION_SCOPE ES LISTA
if isinstance(CONTAMINATION_SCOPE, list):
    cols_to_contaminate = [c for c in CONTAMINATION_SCOPE if c in df.columns]

# DETECTAR COLUMNAS INEXISTENTES
ignored_cols = [c for c in (COLUMNS_TO_CONTAMINATE if not ALL_COLUMNS else []) if c not in df.columns]

# VALIDAR COLUMNAS VÁLIDAS
if not cols_to_contaminate:
    raise ValueError("[ ERROR ] NO HAY COLUMNAS VÁLIDAS PARA CONTAMINAR")

# MOSTRAR INFORMACIÓN DE COLUMNAS
if SHOW_INFO:
    print(f"[ INFO ] COLUMNAS A CONTAMINAR: {len(cols_to_contaminate)}")
    if ignored_cols:
        print(f"[ INFO ] COLUMNAS IGNORADAS: {ignored_cols}")

# SELECCIONAR FILAS A CONTAMINAR
np.random.seed(RANDOM_STATE)
n_rows = df.shape[0]
n_contam = int(n_rows * CONTAMINATION_RATE)
contam_indices = np.random.choice(df.index, size=n_contam, replace=False)
if SHOW_INFO:
    print(f"[ INFO ] FILAS A CONTAMINAR: {n_contam}")

# COPIAR DATASET ORIGINAL
df_contaminated = df.copy()

# APLICAR CONTAMINACIÓN
for col in cols_to_contaminate:
    if df[col].dtype in ['int64', 'float64']:
        noise = np.random.normal(0, NOISE_INTENSITY * df[col].std(), size=n_contam)  # GENERAR RUIDO
        df_contaminated.loc[contam_indices, col] += noise
if SHOW_INFO:
    print(f"[ INFO ] CONTAMINACIÓN APLICADA")

# AÑADIR COLUMNA DE ANOMALÍA
if ADD_LABEL:
    df_contaminated['is_anomaly'] = 0  # INICIALIZAR COMO NORMAL
    df_contaminated.loc[contam_indices, 'is_anomaly'] = 1  # MARCAR FILAS CONTAMINADAS
    if SHOW_INFO:
        print("[ INFO ] COLUMNA 'IS_ANOMALY' AÑADIDA")

# GUARDAR DATASET CONTAMINADO
df_contaminated.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] DATASET CONTAMINADO EN '{OUTPUT_CSV}'")
