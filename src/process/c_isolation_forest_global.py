# ARCHIVO: c_isolation_forest_global.py
# APLICA ISOLATION FOREST AL DATASET COMPLETO PARA DETECCI ON GLOBAL DE ANOMALIAS

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import json

# CARGAR HIPERPARÁMETROS
try:
    with open('hiperparameters.json', 'r') as config_file:
        config = json.load(config_file)['c_isolation_forest_global']
except (FileNotFoundError, KeyError) as e:
    print(f"[ERROR]: NO SE PUDO CARGAR hiperparameters.JSON: {e}")
    exit(1)

# EXTRAER HIPERPARÁMETROS
contaminated_path = config['contaminated_dataset_path']
csv_encoding = config['csv_encoding']
results_dir = config['results_directory']
log_path = config['log_file_path']
output_path = config['output_dataset_path']
temporal_cols = config['temporal_columns']
datetime_col = config['datetime_column']
low_memory = config['low_memory']
contamination = config['contamination_level']
random_state = config['random_state']

# CREAR DIRECTORIO DE RESULTADOS
os.makedirs(results_dir, exist_ok=True)

# INICIAR LOG
with open(log_path, 'a') as log_file:
    log_file.write("\n[ c_isolation_forest_global ]\n")

# FUNCI ON PARA REGISTRAR MENSAJES
def log_message(message):
    message_upper = message.upper()
    print(message_upper)
    with open(log_path, 'a') as log_file:
        log_file.write(message_upper + "\n")

# VERIFICAR EXISTENCIA DEL DATASET
log_message(f"[CARGANDO DATASET]: {contaminated_path}")
if not os.path.exists(contaminated_path):
    log_message(f"[ERROR]: {contaminated_path} NO ENCONTRADO. EJECUTAR b_preliminary_analysis PRIMERO.")
    raise FileNotFoundError(f"{contaminated_path} NO ENCONTRADO.")

# CARGAR DATASET
try:
    with open(contaminated_path, 'r', encoding=csv_encoding, errors='replace') as f:
        dataset = pd.read_csv(f, low_memory=low_memory)
    log_message("[CARGADO]: DATASET CARGADO CON EXITO.")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO CARGAR EL DATASET: {e}")
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

# SELECCIONAR COLUMNAS DE CARACTERISTICAS
feature_cols = dataset.select_dtypes(include=[np.number]).columns.drop(temporal_cols, errors='ignore').tolist()
log_message(f"-> COLUMNAS NUMERICAS SELECCIONADAS: {len(feature_cols)}")
X = dataset[feature_cols]

# APLICAR ISOLATION FOREST
global_if_model = IsolationForest(contamination=contamination, random_state=random_state)
global_anomaly_labels = global_if_model.fit_predict(X)

# AGREGAR ETIQUETAS DE ANOMALIAS
dataset['global_anomaly'] = global_anomaly_labels
anomalies_count = np.sum(global_anomaly_labels == -1)
log_message(f"-> ANOMALIAS GLOBALES DETECTADAS: {anomalies_count}")

# GUARDAR DATASET CON ETIQUETAS
dataset.to_csv(output_path, index=False, encoding=csv_encoding)

# VERIFICAR GUARDADO
if os.path.exists(output_path):
    log_message(f"[GUARDADO]: {output_path}")
else:
    log_message(f"[ERROR]: {output_path} NO GUARDADO.")