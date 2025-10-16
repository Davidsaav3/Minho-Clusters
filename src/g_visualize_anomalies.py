# ARCHIVO: g_visualize_anomalies.PY
# GENERA VISUALIZACIONES DE ANOMALIAS DEL DATASET FINAL USANDO MATPLOTLIB

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import traceback

# CARGAR HIPERPARÁMETROS
try:
    with open('hiperparameters.json', 'r') as config_file:
        config = json.load(config_file)['g_visualize_anomalies']
except (FileNotFoundError, KeyError) as e:
    print(f"[ERROR]: NO SE PUDO CARGAR hiperparameters.json: {e}".upper())
    exit(1)

# EXTRAER HIPERPARÁMETROS
final_path = config['final_dataset_path']
plots_dir = config['plots_directory']
log_path = config['log_file_path']
csv_encoding = config['csv_encoding']
datetime_col = config['datetime_column']
low_memory = config['low_memory']

# CREAR DIRECTORIO DE GRÁFICOS
os.makedirs(plots_dir, exist_ok=True)

# INICIAR LOG
with open(log_path, 'a') as log_file:
    log_file.write("\n[ g_visualize_anomalies ]\n")

# FUNCI ON PARA REGISTRAR MENSAJES
def log_message(message):
    message_upper = message.upper()
    print(message_upper)
    with open(log_path, 'a') as log_file:
        log_file.write(message_upper + "\n")

# VERIFICAR EXISTENCIA DEL DATASET
log_message(f"[CARGANDO DATASET]: {final_path}")
if not os.path.exists(final_path):
    log_message(f"[ERROR]: {final_path} NO ENCONTRADO. EJECUTAR f_isolation_forest_per_cluster PRIMERO.")
    raise FileNotFoundError(f"{final_path} NO ENCONTRADO.")

# CARGAR DATASET
try:
    dataset = pd.read_csv(final_path, encoding=csv_encoding, low_memory=low_memory)
    log_message("[CARGADO]: DATASET CARGADO CON EXITO.")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO CARGAR CON {csv_encoding}: {e}. INTENTANDO ISO-8859-1...")
    try:
        dataset = pd.read_csv(final_path, encoding='ISO-8859-1', low_memory=low_memory)
        log_message("[CARGADO]: DATASET CARGADO CON ISO-8859-1.")
    except Exception as e2:
        log_message(f"[ERROR FATAL]: NO SE PUDO CARGAR EL DATASET: {e2}")
        traceback.print_exc()
        raise

# VERIFICAR COLUMNAS NECESARIAS
missing_cols = [col for col in [datetime_col, 'global_anomaly', 'cluster_anomaly', 'nivel_plaxiquet'] if col not in dataset.columns]
if missing_cols:
    log_message(f"[ERROR]: COLUMNAS FALTANTES: {missing_cols}")
    raise ValueError(f"Columnas faltantes: {missing_cols}")

# RENOMBRAR COLUMNA CON BOM SI EXISTE
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': datetime_col}, inplace=True)
    log_message("-> RENOMBRADA 'ï»¿DATETIME' A 'DATETIME'.")

# CONVERTIR COLUMNA DATETIME
if datetime_col in dataset.columns:
    dataset[datetime_col] = pd.to_datetime(dataset[datetime_col], errors='coerce')
    dataset = dataset.sort_values(datetime_col)
    log_message("-> CONVERSI ON DE DATETIME COMPLETADA.")
else:
    log_message(f"[ADVERTENCIA]: COLUMNA '{datetime_col}' NO ENCONTRADA.")

# CONVERTIR TIPOS DE DATOS
for col in ['global_anomaly', 'cluster_anomaly']:
    if col in dataset.columns:
        dataset[col] = dataset[col].astype(float)
        log_message(f"-> {col} CONVERTIDO A FLOAT.")

# MOSTRAR INFORMACI ON DEL DATASET
log_message(f"-> TAMANYO DEL DATASET: {dataset.shape}")

# 1. GRÁFICO DE BARRAS: ANOMALIAS GLOBALES
if 'global_anomaly' in dataset.columns and 'nivel_plaxiquet' in dataset.columns:
    log_message(f"-> VALORES ÚNICOS EN 'global_anomaly': {dataset['global_anomaly'].unique()}")
    log_message(f"-> NULOS EN 'nivel_plaxiquet': {dataset['nivel_plaxiquet'].isnull().sum()}")
    levels = sorted(dataset['nivel_plaxiquet'].dropna().unique().astype(str))
    if levels:
        normal_val, anomaly_val = 1.0, -1.0
        normals = dataset[dataset['global_anomaly'] == normal_val]['nivel_plaxiquet'].value_counts().reindex(levels, fill_value=0)
        anomalies = dataset[dataset['global_anomaly'] == anomaly_val]['nivel_plaxiquet'].value_counts().reindex(levels, fill_value=0)
        if normals.sum() == 0 and anomalies.sum() == 0:
            log_message(": SIN DATOS PARA NORMALES Y ANOMALIAS EN 'global_anomaly'. OMITIENDO GRÁFICO.")
        else:
            x = np.arange(len(levels))
            width = 0.35
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, normals, width, label='Normal', color='blue')
            ax.bar(x + width/2, anomalies, width, label='AnomalIa', color='red')
            ax.set_title('NORMALES Y ANOMALIAS GLOBALES POR NIVEL DE PLAXIQUET')
            ax.set_xlabel('NIVEL DE PLAXIQUET')
            ax.set_ylabel('NÚMERO DE OBSERVACIONES')
            ax.set_xticks(x)
            ax.set_xticklabels(levels, rotation=45, ha='right')
            ax.legend()
            plot_path = os.path.join(plots_dir, 'global_normals_anomalies_by_plaxiquet.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            log_message(f"[GUARDADO]: {plot_path}")
    else:
        log_message(": SIN NIVELES EN 'nivel_plaxiquet'. OMITIENDO GRÁFICO.")
else:
    log_message(": FALTAN COLUMNAS PARA GRÁFICO GLOBAL.")

# 2. GRÁFICO DE BARRAS: ANOMALIAS POR CLÚSTER
if 'cluster_anomaly' in dataset.columns and 'nivel_plaxiquet' in dataset.columns:
    log_message(f"-> VALORES ÚNICOS EN 'cluster_anomaly': {dataset['cluster_anomaly'].unique()}")
    levels = sorted(dataset['nivel_plaxiquet'].dropna().unique().astype(str))
    if levels:
        normal_val, anomaly_val = 1.0, -1.0
        normals = dataset[dataset['cluster_anomaly'] == normal_val]['nivel_plaxiquet'].value_counts().reindex(levels, fill_value=0)
        anomalies = dataset[dataset['cluster_anomaly'] == anomaly_val]['nivel_plaxiquet'].value_counts().reindex(levels, fill_value=0)
        if normals.sum() == 0 and anomalies.sum() == 0:
            log_message(": SIN DATOS PARA NORMALES Y ANOMALIAS EN 'cluster_anomaly'. OMITIENDO GRÁFICO.")
        else:
            x = np.arange(len(levels))
            width = 0.35
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, normals, width, label='Normal', color='blue')
            ax.bar(x + width/2, anomalies, width, label='AnomalIa', color='red')
            ax.set_title('NORMALES Y ANOMALIAS POR CLÚSTER POR NIVEL DE PLAXIQUET')
            ax.set_xlabel('NIVEL DE PLAXIQUET')
            ax.set_ylabel('NÚMERO DE OBSERVACIONES')
            ax.set_xticks(x)
            ax.set_xticklabels(levels, rotation=45, ha='right')
            ax.legend()
            plot_path = os.path.join(plots_dir, 'cluster_normals_anomalies_by_plaxiquet.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            log_message(f"[GUARDADO]: {plot_path}")
    else:
        log_message(": SIN NIVELES EN 'nivel_plaxiquet'. OMITIENDO GRÁFICO.")
else:
    log_message(": FALTAN COLUMNAS PARA GRÁFICO DE CLÚSTER.")