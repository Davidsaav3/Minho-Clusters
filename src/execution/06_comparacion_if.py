# src/06_comparacion_if.py
import pandas as pd
import glob
import os

# =========================
# Funciones auxiliares
# =========================

def secuencia_info(series):
    """Cuenta secuencias consecutivas de 1s y devuelve la cantidad y longitud máxima."""
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

results_path = '../../results/execution'
files = glob.glob(os.path.join(results_path, '*.csv'))

# Buscar IF global
global_files = [f for f in files if 'if_global' in os.path.basename(f)]
if not global_files:
    raise FileNotFoundError("No se encontró ningún archivo que contenga 'if_global' en ../results/")
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
# Definir agrupaciones de clusters
# =========================
cluster_groups = {
    "cluster-temporal": ["cluster-temporal_Date_and_Time", "cluster-temporal_Calendar"],
    "cluster-ubicacion": [
        "cluster-ubicacion_Pozo", "cluster-ubicacion_Falconera", "cluster-ubicacion_Ull_Pueblo",
        "cluster-ubicacion_Playa", "cluster-ubicacion_Beniopa", "cluster-ubicacion_Llombart",
        "cluster-ubicacion_Sanjuan"
    ],
    "cluster-medida": [
        "cluster-medida_Calidad_Agua", "cluster-medida_Hidraulica", "cluster-medida_Operacional"
    ],
    "cluster-ambientales": [
        "cluster-ambientales_Climatologia", "cluster-ambientales_Season_and_Calendar"
    ],
    "cluster-eficiencia": [
        "cluster-eficiencia_Consumo_Electrico", "cluster-eficiencia_Funcionamiento_y_Rendimiento",
        "cluster-eficiencia_Estado_del_Pozo"
    ],
    "cluster-combinado": [
        "cluster-combinado_Temporal_y_Flujo", "cluster-combinado_Calidad_y_Ubicacion",
        "cluster-combinado_Operacional_y_Clima"
    ]
}

# =========================
# Evaluar y mostrar resultados agrupados
# =========================
for grupo, subclusters in cluster_groups.items():
    print(f"\n=== {grupo.upper()} ===")
    for sub in subclusters:
        file_path = os.path.join(results_path, f"{sub}.csv")
        if not os.path.exists(file_path):
            print(f"  [!] No se encontró archivo: {sub}.csv")
            continue

        df_cluster = pd.read_csv(file_path)

        # Detectar columna de anomalías
        if 'anomaly' in df_cluster.columns:
            anomaly_col = 'anomaly'
        elif 'anomaly_global' in df_cluster.columns:
            anomaly_col = 'anomaly_global'
        else:
            print(f"  [!] No se encuentra columna de anomalías en {sub}")
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
        porcentaje = 100 * coincidencias / total_cluster if total_cluster > 0 else 0

        print(f"  {sub}:")
        print(f"    Anomalías detectadas: {total_cluster}")
        print(f"    Secuencias de anomalías: {seq_count}, Longitud máxima: {seq_max}")
        print(f"    Coinciden con IF global: {coincidencias} ({porcentaje:.2f}%)")
        print("    " + "-"*40)

print("\n=== PIPELINE COMPLETO ===")
print("Todos los resultados intermedios y finales están en '../results/'")
