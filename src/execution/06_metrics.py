import pandas as pd  # MANEJO DE DATAFRAMES
import glob          # BUSCAR ARCHIVOS POR PATRÓN
import os            # MANEJO DE RUTAS Y CREACIÓN DE CARPETAS
import json          # GUARDAR RESULTADOS EN JSON

# PARÁMETROS CONFIGURABLES
RESULTS_FOLDER = '../../results/execution'                       # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN
GLOBAL_FILE_PATTERN = 'if_global'                     # PATRÓN PARA IDENTIFICAR IF GLOBAL
CLUSTERS_JSON = 'clusters.json'                       # ARCHIVO JSON CON DEFINICIÓN DE CLUSTERS
OUTPUT_JSON = os.path.join(RESULTS_FOLDER, '06_results.json') # JSON FINAL DE RESULTADOS
SHOW_INFO = True                                      # MOSTRAR INFORMACIÓN EN PANTALLA

# CREAR CARPETAS NECESARIAS
os.makedirs(RESULTS_FOLDER, exist_ok=True)            # CREAR CARPETA PRINCIPAL SI NO EXISTE
os.makedirs(EXECUTION_FOLDER, exist_ok=True)         # CREAR CARPETA DE EJECUCIÓN SI NO EXISTE
if SHOW_INFO:
    print(f"[ INFO ] Carpetas creadas si no existían")

# CARGAR IF GLOBAL
files = glob.glob(os.path.join(EXECUTION_FOLDER, '*.csv'))  # LISTAR TODOS LOS CSV
global_files = [f for f in files if GLOBAL_FILE_PATTERN in os.path.basename(f)]  # FILTRAR IF GLOBAL
if not global_files:
    raise FileNotFoundError(f"[ ERROR ] No se encontró archivo con patrón '{GLOBAL_FILE_PATTERN}'")

df_global = pd.read_csv(global_files[0])  # CARGAR CSV GLOBAL

# DETECTAR COLUMNAS IMPORTANTES
if 'anomaly' in df_global.columns:
    anomaly_col = 'anomaly'  # COLUMNA DE ANOMALÍAS GLOBALES
elif 'anomaly' in df_global.columns:
    anomaly_col = 'anomaly'
else:
    raise ValueError("[ ERROR ] No se encontró columna de anomalías en IF global")

if 'sequence' in df_global.columns:
    sequence_col = 'sequence'  # COLUMNA DE SECUENCIA
else:
    raise ValueError("[ ERROR ] No existe columna 'sequence' en IF global")

# RESUMEN IF GLOBAL
total_global = df_global[anomaly_col].sum()  # CALCULAR TOTAL DE ANOMALÍAS GLOBALES
max_sequence_global = df_global[sequence_col].max() # CALCULAR LONGITUD MÁXIMA DE SECUENCIA
results = {'if_global': {'total_anomalies': int(total_global),
                         'max_sequence': int(max_sequence_global)},
           'clusters': {}}

# CARGAR DEFINICIÓN DE CLUSTERS
with open(CLUSTERS_JSON, 'r', encoding='utf-8') as f:
    clusters_json = json.load(f)  # CARGAR CONFIGURACIÓN DE CLUSTERS DESDE JSON
if SHOW_INFO:
    print(f"[ INFO ] Clusters cargados desde '{CLUSTERS_JSON}'")

# CONSTRUIR GRUPOS DE ARCHIVOS POR CLUSTER
cluster_groups = {}
for cluster_name, subparts in clusters_json.items():
    group_key = f"cluster_{cluster_name}"  # NOMBRE DEL GRUPO
    cluster_groups[group_key] = [f"cluster_{cluster_name}_{sub_name}" for sub_name in subparts.keys()]  # ARCHIVOS POR SUBCLUSTER

# EVALUAR CADA CLUSTER
for grupo, subclusters in cluster_groups.items():
    results['clusters'][grupo] = {}
    for sub in subclusters:
        file_path = os.path.join(EXECUTION_FOLDER, f"{sub}.csv")  # RUTA DEL CSV
        if not os.path.exists(file_path):
            results['clusters'][grupo][sub] = {'error': 'archivo no encontrado'}  # MARCAR ERROR SI NO EXISTE
            continue

        df_cluster = pd.read_csv(file_path)  # CARGAR CSV DEL CLUSTER

        # DETECTAR COLUMNA DE ANOMALÍAS
        if 'anomaly' in df_cluster.columns:
            anomaly_col = 'anomaly'
        elif 'anomaly' in df_cluster.columns:
            anomaly_col = 'anomaly'
        else:
            results['clusters'][grupo][sub] = {'error': 'columna de anomalías no encontrada'}
            continue

        # VERIFICAR QUE EXISTE LA COLUMNA 'SEQUENCE'
        if 'sequence' not in df_cluster.columns:
            results['clusters'][grupo][sub] = {'error': 'columna sequence no encontrada'}
            continue

        # CALCULAR TOTAL DE ANOMALÍAS Y MÁXIMA SECUENCIA DEL CLUSTER
        total_cluster = df_cluster[anomaly_col].sum()
        max_sequence_cluster = df_cluster['sequence'].max()

        # CALCULAR COINCIDENCIAS CON IF GLOBAL
        merged = pd.merge(
            df_global[[anomaly_col]].rename(columns={anomaly_col:'global_anomaly'}),
            df_cluster[[anomaly_col]].rename(columns={anomaly_col:'cluster_anomaly'}),
            left_index=True, right_index=True
        )
        coincidencias = ((merged['global_anomaly']==1) & (merged['cluster_anomaly']==1)).sum()
        porcentaje = 100 * coincidencias / total_cluster if total_cluster > 0 else 0

        # GUARDAR RESULTADOS DEL CLUSTER
        results['clusters'][grupo][sub] = {
            'total_anomalies': int(total_cluster),
            'max_sequence': int(max_sequence_cluster),
            'coincidences_with_global': int(coincidencias),
            'coincidence_percent': round(porcentaje, 2)
        }

# GUARDAR RESULTADOS EN JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)  # GUARDAR RESULTADOS FINALES EN JSON
if SHOW_INFO:
    print(f"[ GUARDADO ] Resultados de IF global y clusters en '{OUTPUT_JSON}'")

# PRECISI ON: 0.5081
# RECALL: 0.4829
# F1-SCORE: 0.4952
# EXACTITUD: 0.9482
# MCC: 0.4681
# ANOMALIAS REALES: 3289
# ANOMALIAS DETECTADAS: 7018
# DETECCIONES CORRECTAS: 3566
# FALSOS POSITIVOS: 3452
# FALSOS NEGATIVOS: 3818
# RATIO DE DETECCI ON: 0.4829
# RATIO DE FALSOS POSITIVOS: 0.3297