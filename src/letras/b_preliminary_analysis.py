# ARCHIVO: b_preliminary_analysis.PY
# REALIZA ANÁLISIS PRELIMINAR: VERIFICA LIMPIEZA, APLICA ISOLATION FOREST Y GENERA RESÚMENES

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings
import os
import json
import traceback
warnings.filterwarnings("ignore")

# CARGAR HIPERPARÁMETROS
try:
    with open('hiperparameters.json', 'r') as config_file:
        config = json.load(config_file)['b_preliminary_analysis']
except (FileNotFoundError, KeyError) as e:
    print(f"[ERROR]: NO SE PUDO CARGAR hiperparameters.json: {e}".upper())
    exit(1)

# EXTRAER HIPERPARÁMETROS
clean_path = config['clean_dataset_path']
results_dir = config['results_directory']
log_path = config['log_file_path']
feature_columns_path = config['feature_columns_path']
adjusted_path = config['adjusted_anomalies_path']
stats_path = config['stats_summary_path']
critical_path = config['critical_columns_path']
contaminated_path = config['contaminated_dataset_path']
csv_encoding = config['csv_encoding']
datetime_col = config['datetime_column']
temporal_cols = config['temporal_columns']
low_memory = config['low_memory']
initial_contamination = config['initial_contamination']
adjusted_contamination = config['adjusted_contamination']
random_seed = config['random_seed']
artificial_contamination_fraction = config['artificial_contamination_fraction']
temp_anomaly_low = config['temp_anomaly_low']
temp_anomaly_high = config['temp_anomaly_high']
fallback_anomaly_low = config['fallback_anomaly_low']
fallback_anomaly_high = config['fallback_anomaly_high']
summary_num_features = config['summary_num_features']
extreme_threshold = config['extreme_threshold']

# CREAR DIRECTORIO DE RESULTADOS
os.makedirs(results_dir, exist_ok=True)

# INICIAR LOG
with open(log_path, 'a') as log_file:
    log_file.write("\n[ b_preliminary_analysis ]\n")

# FUNCI ON PARA REGISTRAR MENSAJES
def log_message(message):
    message_upper = message.upper()
    print(message_upper)
    with open(log_path, 'a') as log_file:
        log_file.write(message_upper + "\n")

# VERIFICAR EXISTENCIA DEL DATASET
if not os.path.exists(clean_path):
    log_message(f"[ERROR]: {clean_path} NO ENCONTRADO. EJECUTAR a_data_preparation PRIMERO.")
    raise FileNotFoundError(f"{clean_path} NO ENCONTRADO.")

# CARGAR DATASET
try:
    dataset = pd.read_csv(clean_path, encoding=csv_encoding, low_memory=low_memory)
    log_message("[CARGADO]: DATASET CARGADO CON EXITO.")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO CARGAR CON {csv_encoding}: {e}. INTENTANDO ISO-8859-1...")
    try:
        dataset = pd.read_csv(clean_path, encoding='ISO-8859-1', low_memory=low_memory)
        log_message("[CARGADO]: DATASET CARGADO CON ISO-8859-1.")
    except Exception as e2:
        log_message(f"[ERROR FATAL]: NO SE PUDO CARGAR EL DATASET: {e2}")
        traceback.print_exc()
        raise

# VERIFICAR NULOS
log_message("-> DATASET SIN NULOS." if dataset.isnull().sum().sum() == 0 else "[ADVERTENCIA]: DATASET CON NULOS.")

# SELECCIONAR COLUMNAS NUMERICAS
feature_columns = dataset.select_dtypes(include=[np.number]).columns.drop(temporal_cols, errors='ignore').tolist()
log_message(f"-> COLUMNAS NUMERICAS SELECCIONADAS: {len(feature_columns)}")

# GUARDAR COLUMNAS SELECCIONADAS
try:
    pd.DataFrame({'feature_columns': feature_columns}).to_csv(feature_columns_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {feature_columns_path}")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO GUARDAR {feature_columns_path}: {e}")
    raise

# APLICAR ISOLATION FOREST
X = dataset[feature_columns]
if_model = IsolationForest(contamination=initial_contamination, random_state=random_seed)
anomaly_labels = if_model.fit_predict(X)
natural_anomalies_count = np.sum(anomaly_labels == -1)
log_message(f"-> ANOMALIAS NATURALES DETECTADAS: {natural_anomalies_count}")

# INTRODUCIR CONTAMINACI ON ARTIFICIAL
np.random.seed(random_seed)
artificial_indices = np.random.choice(X.index, size=int(artificial_contamination_fraction * len(X)), replace=False)
X_artificial = X.copy()
if 'aemet_temperatura_media' in X_artificial.columns:
    X_artificial.loc[artificial_indices, 'aemet_temperatura_media'] += np.random.uniform(temp_anomaly_low, temp_anomaly_high, size=len(artificial_indices))
else:
    first_col = feature_columns[0] if feature_columns else None
    if first_col:
        X_artificial.loc[artificial_indices, first_col] += np.random.uniform(fallback_anomaly_low, fallback_anomaly_high, size=len(artificial_indices))
    else:
        log_message("[ADVERTENCIA]: NO HAY COLUMNAS NUMERICAS PARA CONTAMINACI ON.")

# APLICAR ISOLATION FOREST A DATOS CONTAMINADOS
if_model_adjusted = IsolationForest(contamination=adjusted_contamination, random_state=random_seed)
anomaly_labels_adjusted = if_model_adjusted.fit_predict(X_artificial)
adjusted_anomalies_count = np.sum(anomaly_labels_adjusted == -1)
log_message(f"-> ANOMALIAS AJUSTADAS DETECTADAS: {adjusted_anomalies_count}")

# GUARDAR CONTEO DE ANOMALIAS AJUSTADAS
try:
    pd.DataFrame({'adjusted_anomalies_count': [adjusted_anomalies_count]}).to_csv(adjusted_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {adjusted_path}")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO GUARDAR {adjusted_path}: {e}")
    raise

# RESUMEN ESTADISTICO
stats_summary = X[feature_columns[:summary_num_features]].describe().loc[['min', 'max', 'mean', '50%']]
try:
    stats_summary.to_csv(stats_path, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {stats_path}")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO GUARDAR {stats_path}: {e}")
    raise

# IDENTIFICAR COLUMNAS CRITICAS
critical_columns = [col for col in feature_columns if pd.notna(X[col].max()) and pd.notna(X[col].mean()) and 
                    (X[col].max() > extreme_threshold * X[col].mean() or X[col].min() < X[col].mean() / extreme_threshold)]
try:
    pd.DataFrame({'critical_columns': critical_columns}).to_csv(critical_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {critical_path}")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO GUARDAR {critical_path}: {e}")
    raise

# GUARDAR DATASET CONTAMINADO
try:
    X_artificial_df = dataset.copy()
    X_artificial_df[feature_columns] = X_artificial
    X_artificial_df.to_csv(contaminated_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {contaminated_path}")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO GUARDAR {contaminated_path}: {e}")
    raise