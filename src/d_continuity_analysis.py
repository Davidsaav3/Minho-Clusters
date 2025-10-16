# ARCHIVO: d_continuity_analysis.PY
# ANALIZA CONTINUIDAD DE ANOMALIAS ASIGNANDO ORDEN DE SECUENCIA Y EVALUANDO PERSISTENCIA

import pandas as pd
import numpy as np
import os
import json

# CARGAR HIPERPARÁMETROS
try:
    with open('hiperparameters.json', 'r') as config_file:
        config = json.load(config_file)['d_continuity_analysis']
except (FileNotFoundError, KeyError) as e:
    print(f"[ERROR]: NO SE PUDO CARGAR hiperparameters.JSON: {e}".upper())
    exit(1)

# EXTRAER HIPERPARÁMETROS
input_path = config['input_dataset_path']
csv_encoding = config['csv_encoding']
results_dir = config['results_directory']
log_path = config['log_file_path']
output_path = config['output_dataset_path']
coherence_summary_path = config['coherence_summary_path']
datetime_col = config['datetime_column']
low_memory = config['low_memory']
anomaly_col = config['anomaly_label_column']

# CREAR DIRECTORIO DE RESULTADOS
os.makedirs(results_dir, exist_ok=True)

# INICIAR LOG
with open(log_path, 'a') as log_file:
    log_file.write("\n[ d_continuity_analysis ]\n")

# FUNCI ON PARA REGISTRAR MENSAJES
def log_message(message):
    message_upper = message.upper()
    print(message_upper)
    with open(log_path, 'a') as log_file:
        log_file.write(message_upper + "\n")

# VERIFICAR EXISTENCIA DEL DATASET
log_message(f"[CARGANDO DATASET]: {input_path}")
if not os.path.exists(input_path):
    log_message(f"[ERROR]: {input_path} NO ENCONTRADO. EJECUTAR c_isolation_forest_global PRIMERO.")
    raise FileNotFoundError(f"{input_path} NO ENCONTRADO.")

# CARGAR DATASET
try:
    with open(input_path, 'r', encoding=csv_encoding, errors='replace') as f:
        dataset = pd.read_csv(f, low_memory=low_memory)
    log_message("[CARGADO]: DATASET CARGADO CON EXITO.")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO CARGAR EL DATASET: {e}")
    raise

# VERIFICAR COLUMNA DE ANOMALIAS
if anomaly_col not in dataset.columns:
    log_message(f"[ERROR]: COLUMNA DE ANOMALIAS '{anomaly_col}' NO ENCONTRADA.")
    raise ValueError(f"Columna '{anomaly_col}' no encontrada.")

# RENOMBRAR COLUMNA CON BOM SI EXISTE
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': datetime_col}, inplace=True)
    log_message("-> RENOMBRADA 'ï»¿DATETIME' A 'DATETIME'.")

# CONVERTIR COLUMNA DATETIME
if datetime_col in dataset.columns:
    dataset[datetime_col] = pd.to_datetime(dataset[datetime_col], errors='coerce')
    log_message("-> CONVERSI ON DE DATETIME COMPLETADA.")
else:
    log_message(f"[ADVERTENCIA]: COLUMNA '{datetime_col}' NO ENCONTRADA.")

# ORDENAR POR DATETIME
if datetime_col in dataset.columns:
    dataset = dataset.sort_values(datetime_col).reset_index(drop=True)
    log_message("-> DATASET ORDENADO POR DATETIME.")
else:
    log_message("[ADVERTENCIA]: COLUMNA 'DATETIME' NO ENCONTRADA; ORDENANDO POR INDICE.")
    dataset = dataset.sort_index().reset_index(drop=True)
    log_message("-> DATASET ORDENADO POR INDICE.")

# ASIGNAR SECUENCIAS DE ANOMALIAS
anomaly_sequence = []
sequence_counter = 0
for label in dataset[anomaly_col]:
    if label == -1:
        sequence_counter += 1
        anomaly_sequence.append(sequence_counter)
    else:
        anomaly_sequence.append(0)
dataset['anomaly_sequence'] = anomaly_sequence
log_message("-> SECUENCIAS DE ANOMALIAS ASIGNADAS.")

# EVALUAR PERSISTENCIA
dataset['is_persistent'] = (dataset['anomaly_sequence'] > 0) & (dataset['anomaly_sequence'].shift(1) > 0)
persistence_count = dataset['is_persistent'].sum()
log_message(f"-> SECUENCIAS DE ANOMALIAS PERSISTENTES: {persistence_count}")

# GUARDAR RESUMEN DE COHERENCIA TEMPORAL
coherence_summary = dataset[dataset['anomaly_sequence'] > 0].groupby('anomaly_sequence').size()
if not coherence_summary.empty:
    coherence_summary_df = pd.DataFrame({
        'anomaly_sequence': coherence_summary.index,
        'count': coherence_summary.values
    })
    coherence_summary_df.to_csv(coherence_summary_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {coherence_summary_path}")
else:
    log_message(f"[ADVERTENCIA]: NO SE DETECTARON SECUENCIAS DE ANOMALIAS PARA GUARDAR EN {coherence_summary_path}")

# GUARDAR DATASET CON RESULTADOS
dataset.to_csv(output_path, index=False, encoding=csv_encoding)
log_message(f"[GUARDADO]: {output_path}" if os.path.exists(output_path) else f"[ERROR]: {output_path} NO GUARDADO.")