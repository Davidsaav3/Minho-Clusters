# ARCHIVO: f_isolation_forest_per_cluster.PY
# APLICA ISOLATION FOREST POR CLUSTER Y COMPARA ANOMALIAS

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import defaultdict
import os
import json
import traceback

# CARGAR HIPERPARÁMETROS
try:
    with open('hiperparameters.json', 'r') as config_file:
        config = json.load(config_file)['f_isolation_forest_per_cluster']
except (FileNotFoundError, KeyError) as e:
    print(f"[ERROR]: NO SE PUDO CARGAR hiperparameters.JSON: {e}".upper())
    exit(1)

# EXTRAER HIPERPARÁMETROS
clustered_path = config['clustered_dataset_path']
csv_encoding = config['csv_encoding']
results_dir = config['results_directory']
log_path = config['log_file_path']
final_path = config['final_dataset_path']
season_concentration_path = config['season_concentration_path']
cluster_concentration_path = config['cluster_concentration_path']
temporal_cols = config['temporal_columns']
datetime_col = config['datetime_column']
low_memory = config['low_memory']
contamination = config['contamination_level']
random_state = config['random_state']
min_cluster_size = config['min_cluster_size']
temp_col = config['temperature_column']
temp_quantile = config['temperature_quantile']
season_col = config['season_column']
cluster_col = config['cluster_column']

# CREAR DIRECTORIO DE RESULTADOS
os.makedirs(results_dir, exist_ok=True)

# INICIAR LOG
with open(log_path, 'a') as log_file:
    log_file.write("\n[ f_isolation_forest_per_cluster ]\n")

# FUNCI ON PARA REGISTRAR MENSAJES
def log_message(message, print_to_console=True):
    message_upper = message.upper()
    if print_to_console:
        print(message_upper)
    with open(log_path, 'a') as log_file:
        log_file.write(message_upper + "\n")

# VERIFICAR EXISTENCIA DEL DATASET
log_message(f"[CARGANDO DATASET]: {clustered_path}")
if not os.path.exists(clustered_path):
    log_message(f"[ERROR]: {clustered_path} NO ENCONTRADO. EJECUTAR e_clustering PRIMERO.")
    raise FileNotFoundError(f"{clustered_path} NO ENCONTRADO.")

# CARGAR DATASET
try:
    with open(clustered_path, 'r', encoding=csv_encoding, errors='replace') as f:
        dataset = pd.read_csv(f, low_memory=low_memory)
    log_message("[CARGADO]: DATASET CARGADO CON EXITO.")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO CARGAR CON {csv_encoding}: {e}. INTENTANDO ISO-8859-1...")
    try:
        with open(clustered_path, 'r', encoding='ISO-8859-1', errors='replace') as f:
            dataset = pd.read_csv(f, low_memory=low_memory)
        log_message("[CARGADO]: DATASET CARGADO CON ISO-8859-1.")
    except Exception as e2:
        log_message(f"[ERROR FATAL]: NO SE PUDO CARGAR EL DATASET: {e2}")
        traceback.print_exc()
        raise

# VERIFICAR COLUMNAS NECESARIAS
missing_cols = [col for col in [datetime_col, season_col, cluster_col, 'global_anomaly'] if col not in dataset.columns]
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
    log_message("-> CONVERSI ON DE DATETIME COMPLETADA.")
else:
    log_message(f"[ADVERTENCIA]: COLUMNA '{datetime_col}' NO ENCONTRADA.")

# MOSTRAR INFORMACI ON DEL DATASET
log_message(f"-> TAMANYO DEL DATASET: {dataset.shape}")

# SELECCIONAR COLUMNAS DE CARACTERISTICAS
feature_cols = dataset.select_dtypes(include=[np.number]).columns.drop(temporal_cols, errors='ignore').tolist()
log_message(f"-> COLUMNAS NUMERICAS SELECCIONADAS: {len(feature_cols)}")

# AGRUPAR POR CLUSTER
clusters = dataset.groupby(cluster_col)
log_message(f"-> NÚMERO DE CLUSTERS: {len(clusters)}")

# OBTENER TAMANYOS DE CLUSTERS
cluster_sizes = clusters.size()

# PROCESAR CLUSTERS
cluster_anomalies = defaultdict(list)
coincident_anomalies = []
for i, (cluster_name, cluster_data) in enumerate(clusters):
    log_message(f"-> PROCESANDO CLUSTER {i+1}/{len(clusters)}: {cluster_name} (TAMANYO: {len(cluster_data)})", print_to_console=False)
    if len(cluster_data) < min_cluster_size:
        log_message(f"-> OMITIENDO CLUSTER PEQUENYO {cluster_name}")
        continue
    
    try:
        X_cluster = cluster_data[feature_cols]
        #log_message(f"-> X_CLUSTER TAMANYO: {X_cluster.shape}", print_to_console=False)
        
        # APLICAR ISOLATION FOREST
        cluster_if = IsolationForest(contamination=contamination, random_state=random_state)
        cluster_labels = cluster_if.fit_predict(X_cluster)
        #log_message(f"-> CLUSTER {cluster_name}: ANOMALIAS DETECTADAS: {np.sum(cluster_labels == -1)}", print_to_console=False)
        
        # GUARDAR INDICES DE ANOMALIAS
        anomaly_indices = cluster_data.index[cluster_labels == -1].tolist()
        cluster_anomalies[cluster_name] = anomaly_indices
        
        # COMPARAR CON GLOBAL
        global_anoms = cluster_data[cluster_data['global_anomaly'] == -1].index.tolist()
        coincident = set(anomaly_indices) & set(global_anoms)
        coincident_anomalies.extend(list(coincident))
        #log_message(f"-> ANOMALIAS COINCIDENTES EN {cluster_name}: {len(coincident)}")
    except Exception as e:
        log_message(f"[ERROR]: ERROR EN CLUSTER {cluster_name}: {e}")
        log_message("-> TRACEBACK COMPLETO:")
        log_message(traceback.format_exc())
        continue

# CONTAR ANOMALIAS COINCIDENTES
log_message(f"-> TOTAL ANOMALIAS COINCIDENTES: {len(coincident_anomalies)}")

# GUARDAR CONCENTRACI ON POR ESTACI ON
coincident_df = dataset.loc[coincident_anomalies]
concentration_by_season = coincident_df[season_col].value_counts()
if not concentration_by_season.empty:
    season_concentration_df = pd.DataFrame({
        'season': concentration_by_season.index,
        'count': concentration_by_season.values
    })
    season_concentration_df.to_csv(season_concentration_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {season_concentration_path}")
else:
    log_message(f"[ADVERTENCIA]: NO SE DETECTARON ANOMALIAS POR ESTACI ON PARA GUARDAR EN {season_concentration_path}")

# GUARDAR CONCENTRACI ON POR CLUSTER CON TAMANYO
concentration_by_cluster = coincident_df[cluster_col].value_counts()
if not concentration_by_cluster.empty:
    cluster_concentration_df = pd.DataFrame({
        'cluster': concentration_by_cluster.index,
        'count': concentration_by_cluster.values,
        'cluster_size': [cluster_sizes.get(cluster, 0) for cluster in concentration_by_cluster.index]
    })
    cluster_concentration_df.to_csv(cluster_concentration_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {cluster_concentration_path}")
else:
    log_message(f"[ADVERTENCIA]: NO SE DETECTARON ANOMALIAS POR CLUSTER PARA GUARDAR EN {cluster_concentration_path}")

# CARACTERIZAR ANOMALIAS
for season, count in concentration_by_season.head(1).items():
    if temp_col in coincident_df.columns and not coincident_df[temp_col].isnull().all():
        hot_days_anoms = coincident_df[(coincident_df[season_col] == season) & (coincident_df[temp_col] > coincident_df[temp_col].quantile(temp_quantile))]
        if len(hot_days_anoms) > 0:
            log_message(f"-> EJEMPLO: ANOMALIA EN ESTACI ON {season} EN DIAS CALUROSOS ({len(hot_days_anoms)} INSTANCIAS).")
        else:
            log_message(f"-> EJEMPLO: ANOMALIA EN ESTACI ON {season} ({count} INSTANCIAS).")
    else:
        log_message(f"-> EJEMPLO: ANOMALIA EN ESTACI ON {season} ({count} INSTANCIAS).")

# GUARDAR DATASET FINAL
cluster_labels_full = np.ones(len(dataset))
for indices in cluster_anomalies.values():
    cluster_labels_full[indices] = -1
dataset['cluster_anomaly'] = cluster_labels_full
try:
    dataset.to_csv(final_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {final_path}" if os.path.exists(final_path) else f"[ERROR]: {final_path} NO GUARDADO.")
except Exception as e:
    log_message(f"[ERROR FATAL]: NO SE PUDO GUARDAR {final_path}: {e}")
    traceback.print_exc()
    raise