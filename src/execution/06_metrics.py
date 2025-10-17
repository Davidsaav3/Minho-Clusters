import pandas as pd  # MANEJO DE DATAFRAMES
import glob          # BUSCAR ARCHIVOS
import os            # RUTAS Y DIRECTORIOS
import json          # GUARDAR RESULTADOS EN JSON

# FUNCIONES AUXILIARES
def secuencia_info(series):
    """CUENTA SECUENCIAS DE 1s: TOTAL Y LONGITUD MÁXIMA"""
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

# CARGAR ARCHIVOS
results_path = '../../results/execution'
files = glob.glob(os.path.join(results_path, '*.csv'))

# BUSCAR IF GLOBAL
global_files = [f for f in files if 'if_global' in os.path.basename(f)]
if not global_files:
    raise FileNotFoundError("[ ERROR ] No se encontró archivo 'if_global'")
global_file = global_files[0]
df_global = pd.read_csv(global_file)

# DETECTAR COLUMNA DE ANOMALÍAS
if 'anomaly_global' in df_global.columns:
    anomaly_global_col = 'anomaly_global'
elif 'anomaly' in df_global.columns:
    anomaly_global_col = 'anomaly'
else:
    raise ValueError("[ ERROR ] No hay columna de anomalías en IF global")

# IF GLOBAL
total_global = df_global[anomaly_global_col].sum()
seq_count, seq_max = secuencia_info(df_global[anomaly_global_col])
results = {
    'if_global': {
        'total_anomalies': int(total_global),
        'sequence_count': int(seq_count),
        'max_sequence': int(seq_max)
    },
    'clusters': {}
}

# CARGAR DEFINICIÓN DE CLUSTERS DESDE JSON
json_path = 'clusters.json'
with open(json_path, 'r', encoding='utf-8') as f:
    clusters_json = json.load(f)
print(f"[ INFO ] Clusters cargados desde '{json_path}'")

# CONSTRUIR GRUPOS DE ARCHIVOS AUTOMÁTICAMENTE
cluster_groups = {}
for cluster_name, subparts in clusters_json.items():
    group_key = f"cluster_{cluster_name}"
    cluster_groups[group_key] = []
    for sub_name in subparts.keys():
        file_name = f"cluster_{cluster_name}_{sub_name}"
        cluster_groups[group_key].append(file_name)

# EVALUAR CLUSTERS
for grupo, subclusters in cluster_groups.items():
    results['clusters'][grupo] = {}
    for sub in subclusters:
        file_path = os.path.join(results_path, f"{sub}.csv")
        if not os.path.exists(file_path):
            results['clusters'][grupo][sub] = {'error': 'archivo no encontrado'}
            continue

        df_cluster = pd.read_csv(file_path)

        if 'anomaly' in df_cluster.columns:
            anomaly_col = 'anomaly'
        elif 'anomaly_global' in df_cluster.columns:
            anomaly_col = 'anomaly_global'
        else:
            results['clusters'][grupo][sub] = {'error': 'columna de anomalías no encontrada'}
            continue

        total_cluster = df_cluster[anomaly_col].sum()
        seq_count, seq_max = secuencia_info(df_cluster[anomaly_col])

        merged = pd.merge(
            df_global[[anomaly_global_col]].rename(columns={anomaly_global_col:'global_anomaly'}),
            df_cluster[[anomaly_col]].rename(columns={anomaly_col:'cluster_anomaly'}),
            left_index=True, right_index=True
        )
        coincidencias = ((merged['global_anomaly']==1) & (merged['cluster_anomaly']==1)).sum()
        porcentaje = 100 * coincidencias / total_cluster if total_cluster > 0 else 0

        results['clusters'][grupo][sub] = {
            'total_anomalies': int(total_cluster),
            'sequence_count': int(seq_count),
            'max_sequence': int(seq_max),
            'coincidences_with_global': int(coincidencias),
            'coincidence_percent': round(porcentaje,2)
        }

# GUARDAR RESULTADOS EN JSON
os.makedirs('../../results', exist_ok=True)
with open('../../results/06_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("[ GUARDADO ] Resultados de IF global y clusters en '../../results/06_results.json'")
