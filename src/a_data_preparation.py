# ARCHIVO: a_data_preparation.PY
# PREPARA DATASET ELIMINANDO FILAS CON VALORES NULOS EN COLUMNAS NUMERICAS

import pandas as pd
import numpy as np
import os
import json
import traceback

# CARGAR HIPERPARÁMETROS
try:
    with open('hiperparameters.json', 'r') as config_file:
        config = json.load(config_file)['a_data_preparation']
except (FileNotFoundError, KeyError) as e:
    print(f"[ERROR]: NO SE PUDO CARGAR hiperparameters.json: {e}".upper())
    exit(1)

# EXTRAER HIPERPARÁMETROS
input_dataset_path = config['input_dataset_path']
csv_encoding = config['csv_encoding']
results_directory = config['results_directory']
log_file_path = config['log_file_path']
null_counts_output_path = config['null_counts_output_path']
clean_dataset_output_path = config['clean_dataset_output_path']
temporal_columns = config['temporal_columns']
datetime_column = config['datetime_column']
low_memory = config['low_memory']

# CREAR DIRECTORIO DE RESULTADOS
os.makedirs(results_directory, exist_ok=True)

# INICIAR LOG
with open(log_file_path, 'a') as log_file:
    log_file.write("\n[ a_data_preparation ]\n")

# FUNCI ON PARA REGISTRAR MENSAJES
def log_message(message):
    message_upper = message.upper()
    print(message_upper)
    with open(log_file_path, 'a') as log_file:
        log_file.write(message_upper + "\n")

# VERIFICAR EXISTENCIA DEL DATASET
if not os.path.exists(input_dataset_path):
    log_message(f"[ERROR]: {input_dataset_path} NO ENCONTRADO.")
    raise FileNotFoundError(f"{input_dataset_path} NO ENCONTRADO.")

# CARGAR DATASET
try:
    dataset = pd.read_csv(input_dataset_path, encoding=csv_encoding, low_memory=low_memory)
    log_message("[CARGADO]: DATASET CARGADO CON EXITO.")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO CARGAR CON {csv_encoding}: {e}. INTENTANDO ISO-8859-1...")
    try:
        dataset = pd.read_csv(input_dataset_path, encoding='ISO-8859-1', low_memory=low_memory)
        log_message("[CARGADO]: DATASET CARGADO CON ISO-8859-1.")
    except Exception as e2:
        log_message(f"[ERROR FATAL]: NO SE PUDO CARGAR EL DATASET: {e2}")
        traceback.print_exc()
        raise

# RENOMBRAR COLUMNA CON BOM SI EXISTE
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': datetime_column}, inplace=True)
    log_message("-> RENOMBRADA 'ï»¿DATETIME' A 'DATETIME'.")

# CONVERTIR COLUMNA DATETIME
if datetime_column in dataset.columns:
    dataset[datetime_column] = pd.to_datetime(dataset[datetime_column], errors='coerce')
    log_message("-> CONVERSI ON DE DATETIME COMPLETADA.")
else:
    log_message(f"[ADVERTENCIA]: COLUMNA '{datetime_column}' NO ENCONTRADA.")

# MOSTRAR TAMANYO INICIAL
log_message(f"-> TAMANYO INICIAL DEL DATASET: {dataset.shape}")

# DETECTAR VALORES NULOS
null_counts = dataset.isnull().sum()
nulls_to_print = null_counts[null_counts > 0]

# GUARDAR COLUMNAS CON NULOS
try:
    nulls_df = null_counts[null_counts > 0].reset_index()
    nulls_df.columns = ['column_name', 'null_count']
    nulls_df.to_csv(null_counts_output_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {null_counts_output_path}")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO GUARDAR {null_counts_output_path}: {e}")
    raise

# SELECCIONAR COLUMNAS NUMERICAS
numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
log_message(f"-> COLUMNAS NUMERICAS: {len(numeric_cols)}")

# ELIMINAR FILAS CON NULOS EN COLUMNAS NUMERICAS
clean_dataset = dataset.dropna(subset=numeric_cols)
log_message(f"-> TAMANYO DEL DATASET LIMPIO: {clean_dataset.shape}")

# GUARDAR DATASET LIMPIO
try:
    clean_dataset.to_csv(clean_dataset_output_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {clean_dataset_output_path}")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO GUARDAR {clean_dataset_output_path}: {e}")
    raise
