# src/06_comparacion_if.py
import pandas as pd
import glob
import os

# =========================
# Funciones auxiliares
# =========================

def secuencia_info(series):
    """Cuenta secuencias consecutivas de 1s y devuelve la longitud máxima."""
    in_seq = False
    count_seq = 0
    max_len = 0
    current_len = 0
    for v in series:
        if v == 1:
            current_len += 1
            if not in_seq:
                count_seq += 1
                in_seq = True
        else:
            in_seq = False
            current_len = 0
        if current_len > max_len:
            max_len = current_len
    return count_seq, max_len

# =========================
# Cargar archivos
# =========================

results_path = '../results'
files = glob.glob(os.path.join(results_path, '*.csv'))

# Buscar IF global
global_files = [f for f in files if '01_if_global' in os.path.basename(f)]
if not global_files:
    raise FileNotFoundError("No se encontró ningún archivo que contenga '01_if_global' en ../results/")
global_file = global_files[0]
df_global = pd.read_csv(global_file)

# Detectar columna de anomalías en IF global
if 'anomaly_global' in df_global.columns:
    anomaly_global_col = 'anomaly_global'
elif 'anomaly' in df_global.columns:
    anomaly_global_col = 'anomaly'
else:
    raise ValueError("No se encuentra columna de anomalías en IF global")

# =========================
# IF GLOBAL
# =========================
total_global = df_global[anomaly_global_col].sum()
seq_count, seq_max = secuencia_info(df_global[anomaly_global_col])
print("=== IF GLOBAL ===")
print(f"Anomalías detectadas: {total_global}")
print(f"Secuencias de anomalías: {seq_count}, Longitud máxima: {seq_max}")
print('-'*50)

# =========================
# Archivos de clusters
# =========================
cluster_files = [f for f in files if f != global_file and 'cluster-' in os.path.basename(f)]

for file_path in cluster_files:
    df_cluster = pd.read_csv(file_path)
    cluster_name = os.path.basename(file_path).replace('.csv','')
    
    # Detectar columna de anomalías
    if 'anomaly' in df_cluster.columns:
        anomaly_col = 'anomaly'
    elif 'anomaly_global' in df_cluster.columns:
        anomaly_col = 'anomaly_global'
    else:
        print(f"No se encuentra columna de anomalías en {cluster_name}, se salta.")
        continue

    total_cluster = df_cluster[anomaly_col].sum()
    seq_count, seq_max = secuencia_info(df_cluster[anomaly_col])
    
    # Coincidencias con IF global
    merged = pd.merge(
        df_global[[anomaly_global_col]].rename(columns={anomaly_global_col:'global_anomaly'}),
        df_cluster[[anomaly_col]].rename(columns={anomaly_col:'cluster_anomaly'}),
        left_index=True, right_index=True
    )
    coincidencias = ((merged['global_anomaly']==1) & (merged['cluster_anomaly']==1)).sum()
    porcentaje = 100*coincidencias/total_cluster if total_cluster>0 else 0
    
    print(f"{cluster_name}:")
    print(f"  Anomalías detectadas: {total_cluster}")
    print(f"  Secuencias de anomalías: {seq_count}, Longitud máxima: {seq_max}")
    print(f"  Coinciden con IF global: {coincidencias} ({porcentaje:.2f}%)")
    print('-'*50)
