# ARCHIVO: e_clustering.PY
# CREA CLÚSTERES MANUALES BASADOS EN ESTACI ON Y TEMPERATURA

import pandas as pd
import os
import json
import traceback

# CARGAR HIPERPARÁMETROS
try:
    with open('hiperparameters.json', 'r') as config_file:
        config = json.load(config_file)['e_clustering']
except (FileNotFoundError, KeyError) as e:
    print(f"[ERROR]: NO SE PUDO CARGAR HYPERPhiperparametersARAMETERS.JSON: {e}".upper())
    exit(1)

# EXTRAER HIPERPARÁMETROS
input_path = config['input_dataset_path']
csv_encoding = config['csv_encoding']
results_dir = config['results_directory']
log_path = config['log_file_path']
output_path = config['output_dataset_path']
cluster_summary_path = config['cluster_summary_path']
datetime_col = config['datetime_column']
low_memory = config['low_memory']
temp_col = config['temperature_column']
fallback_temp_pattern = config['fallback_temperature_pattern']
season_col = config['season_column']
temp_bins = config['temperature_bins']
bin_labels = config['bin_labels']

# CREAR DIRECTORIO DE RESULTADOS
os.makedirs(results_dir, exist_ok=True)

# INICIAR LOG
with open(log_path, 'a') as log_file:
    log_file.write("\n[ e_clustering ]\n")

# FUNCI ON PARA REGISTRAR MENSAJES
def log_message(message):
    message_upper = message.upper()
    print(message_upper)
    with open(log_path, 'a') as log_file:
        log_file.write(message_upper + "\n")

# VERIFICAR EXISTENCIA DEL DATASET
log_message(f"[CARGANDO DATASET]: {input_path}")
if not os.path.exists(input_path):
    log_message(f"[ERROR]: {input_path} NO ENCONTRADO. EJECUTAR d_continuity_analysis PRIMERO.")
    raise FileNotFoundError(f"{input_path} NO ENCONTRADO.")

# CARGAR DATASET
try:
    with open(input_path, 'r', encoding=csv_encoding, errors='replace') as f:
        dataset = pd.read_csv(f, low_memory=low_memory)
    log_message("[CARGADO]: DATASET CARGADO CON EXITO.")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO CARGAR CON {csv_encoding}: {e}. INTENTANDO ISO-8859-1...")
    try:
        with open(input_path, 'r', encoding='ISO-8859-1', errors='replace') as f:
            dataset = pd.read_csv(f, low_memory=low_memory)
        log_message("[CARGADO]: DATASET CARGADO CON ISO-8859-1.")
    except Exception as e2:
        log_message(f"[ERROR FATAL]: NO SE PUDO CARGAR EL DATASET: {e2}")
        traceback.print_exc()
        raise

# RENOMBRAR COLUMNA CON BOM SI EXISTE
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': datetime_col}, inplace=True)
    log_message("-> RENOMBRADA 'ï»¿DATETIME' A 'DATETIME'.")

# CONVERTIR COLUMNA DATETIME
if datetime_col in dataset.columns:
    dataset[datetime_col] = pd.to_datetime(dataset[datetime_col], errors='coerce')
    log_message("-> CONVERSI ON DE DATETIME")
else:
    log_message(f"[ADVERTENCIA]: COLUMNA '{datetime_col}' NO ENCONTRADA.")

# VERIFICAR COLUMNA DE ESTACI ON
if season_col not in dataset.columns:
    log_message(f"[ERROR]: COLUMNA DE ESTACI ON '{season_col}' NO ENCONTRADA.")
    raise ValueError(f"Columna '{season_col}' no encontrada.")

# BINARIZAR TEMPERATURA PARA CLÚSTERES
if temp_col in dataset.columns and not dataset[temp_col].isnull().all():
    temp_data = dataset[temp_col].dropna()
    if not temp_data.empty:
        temp_bins = pd.cut(dataset[temp_col], bins=temp_bins, labels=bin_labels, duplicates='drop')
        log_message(f"-> TEMPERATURA BINARIZADA USANDO '{temp_col}'.")
    else:
        temp_bins = pd.Series([bin_labels[1]] * len(dataset), index=dataset.index)
        log_message("-> DATOS DE TEMPERATURA VACIOS; USANDO VALOR POR DEFECTO 'MED'.")
else:
    temp_col_fallback = next((col for col in dataset.columns if fallback_temp_pattern.lower() in col.lower()), None)
    if temp_col_fallback and not dataset[temp_col_fallback].isnull().all():
        temp_data = dataset[temp_col_fallback].dropna()
        if not temp_data.empty:
            temp_bins = pd.cut(dataset[temp_col_fallback], bins=temp_bins, labels=bin_labels, duplicates='drop')
            log_message(f"-> TEMPERATURA BINARIZADA USANDO '{temp_col_fallback}' (FALLBACK).")
        else:
            temp_bins = pd.Series([bin_labels[1]] * len(dataset), index=dataset.index)
            log_message("-> DATOS DE TEMPERATURA FALLBACK VACIOS; USANDO VALOR POR DEFECTO 'MED'.")
    else:
        temp_bins = pd.Series([bin_labels[1]] * len(dataset), index=dataset.index)
        log_message("-> NO SE ENCONTR O COLUMNA DE TEMPERATURA; USANDO VALOR POR DEFECTO 'MED'.")

# DEFINIR CLÚSTERES MANUALES
dataset['cluster'] = dataset[season_col].astype(str) + '_' + temp_bins.astype(str)
log_message("-> CLÚSTERES MANUALES DEFINIDOS.")

# GENERAR RESUMEN DE CLÚSTERES
cluster_summary = dataset.groupby('cluster').size()
if not cluster_summary.empty:
    cluster_summary_df = pd.DataFrame({
        'cluster': cluster_summary.index,
        'count': cluster_summary.values
    })
    cluster_summary_df.to_csv(cluster_summary_path, index=False, encoding=csv_encoding)
else:
    log_message(f"[ADVERTENCIA]: NO SE DETECTARON CLÚSTERES PARA GUARDAR EN {cluster_summary_path}")

# GUARDAR DATASET CON CLÚSTERES
try:
    dataset.to_csv(output_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {output_path}" if os.path.exists(output_path) else f"[ERROR]: {output_path} NO GUARDADO.")
except Exception as e:
    log_message(f"[ERROR FATAL]: NO SE PUDO GUARDAR {output_path}: {e}")
    traceback.print_exc()
    raise