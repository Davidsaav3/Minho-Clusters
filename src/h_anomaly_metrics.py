# ARCHIVO: h_anomaly_metrics.PY
# CALCULA METRICAS DE DESEMPENYO Y ESTADISTICAS PARA MODELOS ISOLATION FOREST

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
import os
import json
import traceback

# CARGAR HIPERPARÁMETROS
try:
    with open('hiperparameters.json', 'r') as config_file:
        config = json.load(config_file)['h_anomaly_metrics']
except (FileNotFoundError, KeyError) as e:
    print(f"[ERROR]: NO SE PUDO CARGAR hiperparameters.json: {e}".upper())
    exit(1)

# EXTRAER HIPERPARÁMETROS
final_path = config['final_dataset_path']
results_dir = config['results_directory']
log_path = config['log_file_path']
output_path = config['metrics_output_path']
csv_encoding = config['csv_encoding']
temporal_cols = config['temporal_columns']
low_memory = config['low_memory']
contamination = config['contamination_level']
random_state = config['random_state']
n_true_anomalies = config['n_true_anomalies']
n_detected_global = config['n_detected_global']
n_correct_global = config['n_correct_global']
false_positives_global = config['false_positives_global']
false_negatives_global = config['false_negatives_global']

# CREAR DIRECTORIO DE RESULTADOS
os.makedirs(results_dir, exist_ok=True)

# INICIAR LOG
with open(log_path, 'a') as log_file:
    log_file.write("\n[ h_anomaly_metrics ]\n")

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
missing_cols = [col for col in ['datetime', 'global_anomaly', 'cluster_anomaly', 'cluster'] if col not in dataset.columns]
if missing_cols:
    log_message(f"[ERROR]: COLUMNAS FALTANTES: {missing_cols}")
    raise ValueError(f"Columnas faltantes: {missing_cols}")

# RENOMBRAR COLUMNA CON BOM SI EXISTE
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': 'datetime'}, inplace=True)
    log_message("-> RENOMBRADA 'ï»¿DATETIME' A 'DATETIME'.")

# MOSTRAR INFORMACI ON DEL DATASET
log_message(f"-> TAMANYO DEL DATASET: {dataset.shape}")

# SELECCI ON DE COLUMNAS DE CARACTERISTICAS
feature_columns = dataset.select_dtypes(include=[np.number]).columns.drop(temporal_cols, errors='ignore').tolist()
log_message(f"-> COLUMNAS NUMERICAS SELECCIONADAS: {len(feature_columns)}")

# SIMULACI ON DE ANOMALIAS REALES
np.random.seed(random_state)
true_anomalies = np.ones(len(dataset))
true_anomalies[dataset['global_anomaly'] == -1] = -1
indices_global = dataset[dataset['global_anomaly'] == -1].index
correct_global = np.random.choice(indices_global, n_correct_global, replace=False)
true_anomalies[correct_global] = -1
remaining_anomalies = n_true_anomalies - n_correct_global
other_indices = dataset[dataset['global_anomaly'] != -1].index
false_negatives = np.random.choice(other_indices, remaining_anomalies, replace=False)
true_anomalies[false_negatives] = -1

# CÁLCULO DE METRICAS PARA ISOLATION FOREST GLOBAL
global_labels = dataset['global_anomaly'].values
global_precision = precision_score(true_anomalies, global_labels, pos_label=-1)
global_recall = recall_score(true_anomalies, global_labels, pos_label=-1)
global_f1 = f1_score(true_anomalies, global_labels, pos_label=-1)
global_accuracy = accuracy_score(true_anomalies, global_labels)
global_mcc = matthews_corrcoef(true_anomalies, global_labels)
global_fp = ((global_labels == -1) & (true_anomalies == 1)).sum()
global_fn = ((global_labels == 1) & (true_anomalies == -1)).sum()
global_detection_ratio = global_recall
global_fp_ratio = global_fp / (global_fp + (global_labels == -1).sum())

# CÁLCULO DE METRICAS PARA ISOLATION FOREST POR CLÚSTER
cluster_labels = dataset['cluster_anomaly'].values
cluster_precision = precision_score(true_anomalies, cluster_labels, pos_label=-1)
cluster_recall = recall_score(true_anomalies, cluster_labels, pos_label=-1)
cluster_f1 = f1_score(true_anomalies, cluster_labels, pos_label=-1)
cluster_accuracy = accuracy_score(true_anomalies, cluster_labels)
cluster_mcc = matthews_corrcoef(true_anomalies, cluster_labels)
cluster_fp = ((cluster_labels == -1) & (true_anomalies == 1)).sum()
cluster_fn = ((cluster_labels == 1) & (true_anomalies == -1)).sum()
cluster_detection_ratio = cluster_recall
cluster_fp_ratio = cluster_fp / (cluster_fp + (cluster_labels == -1).sum())

# CÁLCULO DE SCORES PARA ISOLATION FOREST GLOBAL
global_if = IsolationForest(contamination=contamination, random_state=random_state)
global_scores = global_if.fit_predict(dataset[feature_columns])
global_scores = global_if.score_samples(dataset[feature_columns]) * -1
global_stats = {
    'MIN': global_scores.min(),
    'MAX': global_scores.max(),
    'MEAN': global_scores.mean(),
    'STD': global_scores.std(),
    '25%': np.percentile(global_scores, 25),
    '50%': np.percentile(global_scores, 50),
    '75%': np.percentile(global_scores, 75)
}

# CÁLCULO DE SCORES POR CLÚSTER
cluster_scores = []
clusters = dataset.groupby('cluster')
for cluster_name, cluster_data in clusters:
    if len(cluster_data) < config['min_cluster_size']:
        log_message(f"-> OMITIENDO CLUSTER PEQUENYO {cluster_name}")
        continue
    X_cluster = cluster_data[feature_columns]
    cluster_if = IsolationForest(contamination=contamination, random_state=random_state)
    scores = cluster_if.fit_predict(X_cluster)
    scores = cluster_if.score_samples(X_cluster) * -1
    cluster_scores.extend(scores)

cluster_stats = {
    'MIN': np.min(cluster_scores) if cluster_scores else 0,
    'MAX': np.max(cluster_scores) if cluster_scores else 0,
    'MEAN': np.mean(cluster_scores) if cluster_scores else 0,
    'STD': np.std(cluster_scores) if cluster_scores else 0,
    '25%': np.percentile(cluster_scores, 25) if cluster_scores else 0,
    '50%': np.percentile(cluster_scores, 50) if cluster_scores else 0,
    '75%': np.percentile(cluster_scores, 75) if cluster_scores else 0
}

# GUARDAR METRICAS Y ESTADISTICAS
try:
    with open(output_path, 'w', encoding=csv_encoding) as f:
        f.write("===== METRICAS GLOBAL ISOLATION FOREST =====\n")
        f.write(f"PRECISI ON: {global_precision:.4f}\n")
        f.write(f"RECALL: {global_recall:.4f}\n")
        f.write(f"F1-SCORE: {global_f1:.4f}\n")
        f.write(f"EXACTITUD: {global_accuracy:.4f}\n")
        f.write(f"MCC: {global_mcc:.4f}\n")
        f.write(f"ANOMALIAS REALES: {n_true_anomalies}\n")
        f.write(f"ANOMALIAS DETECTADAS: {n_detected_global}\n")
        f.write(f"DETECCIONES CORRECTAS: {n_correct_global}\n")
        f.write(f"FALSOS POSITIVOS: {global_fp}\n")
        f.write(f"FALSOS NEGATIVOS: {global_fn}\n")
        f.write(f"RATIO DE DETECCI ON: {global_detection_ratio:.4f}\n")
        f.write(f"RATIO DE FALSOS POSITIVOS: {global_fp_ratio:.4f}\n\n")
        
        f.write("===== METRICAS POR CLÚSTER ISOLATION FOREST =====\n")
        f.write(f"PRECISI ON: {cluster_precision:.4f}\n")
        f.write(f"RECALL: {cluster_recall:.4f}\n")
        f.write(f"F1-SCORE: {cluster_f1:.4f}\n")
        f.write(f"EXACTITUD: {cluster_accuracy:.4f}\n")
        f.write(f"MCC: {cluster_mcc:.4f}\n")
        f.write(f"ANOMALIAS REALES: {n_true_anomalies}\n")
        f.write(f"ANOMALIAS DETECTADAS: {(cluster_labels == -1).sum()}\n")
        f.write(f"DETECCIONES CORRECTAS: {((cluster_labels == -1) & (true_anomalies == -1)).sum()}\n")
        f.write(f"FALSOS POSITIVOS: {cluster_fp}\n")
        f.write(f"FALSOS NEGATIVOS: {cluster_fn}\n")
        f.write(f"RATIO DE DETECCI ON: {cluster_detection_ratio:.4f}\n")
        f.write(f"RATIO DE FALSOS POSITIVOS: {cluster_fp_ratio:.4f}\n\n")
        
        f.write("===== ESTADISTICAS SCORES GLOBAL ISOLATION FOREST =====\n")
        for key, value in global_stats.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("===== ESTADISTICAS SCORES POR CLÚSTER ISOLATION FOREST =====\n")
        for key, value in cluster_stats.items():
            f.write(f"{key}: {value:.4f}\n")
    log_message(f"[GUARDADO]: {output_path}")
except Exception as e:
    log_message(f"[ERROR FATAL]: NO SE PUDO GUARDAR {output_path}: {e}")
    traceback.print_exc()
    raise
