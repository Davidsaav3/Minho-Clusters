import pandas as pd  # IMPORTAR LIBRERÍA PANDAS
import glob  # IMPORTAR LIBRERÍA GLOB
import os  # IMPORTAR LIBRERÍA OS
import json  # IMPORTAR LIBRERÍA JSON
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef  # IMPORTAR MÉTRICAS DE SKLEARN

# CONFIGURACIÓN DE RUTAS  
RESULTS_FOLDER = '../../results/execution'  # DEFINIR CARPETA DE RESULTADOS
EXECUTION_FOLDER = '../../results/execution'  # DEFINIR CARPETA DE EJECUCIÓN
GLOBAL_FILE_PATTERN = '04_global.csv'  # DEFINIR PATRÓN DE ARCHIVO GLOBAL
CLUSTERS_JSON = 'clusters.json'  # DEFINIR ARCHIVO JSON DE CLUSTERS
OUTPUT_CSV = os.path.join(RESULTS_FOLDER, '06_results.csv')  # DEFINIR RUTA DE SALIDA CSV
INTERMEDIATES_CSV = os.path.join(RESULTS_FOLDER, '06_results_ri_ii.csv')  # DEFINIR RUTA DE INTERMEDIOS CSV
SHOW_INFO = True  # ACTIVAR MOSTRAR INFORMACIÓN

# ORDEN DE COLUMNAS EN CSV FINAL  
columns_order = [  # INICIAR LISTA DE COLUMNAS
    'file', 'anomalies_real', 'anomalies_detected', 'detections_correct', 'false_positives', 'false_negatives',  # PRIMERA PARTE DE COLUMNAS
    'total_sequences', 'max_sequence', 'precision', 'recall', 'f1_score', 'accuracy', 'mcc',  # SEGUNDA PARTE DE COLUMNAS
    'ratio_detection', 'ratio_fp', 'perc_global_anomalies_detected', 'perc_cluster_vs_global', 'total_coincidences'  # TERCERA PARTE DE COLUMNAS
]

# CARGAR IF GLOBAL  
files = glob.glob(os.path.join(EXECUTION_FOLDER, '*.csv'))           # LISTAR TODOS LOS CSV EN EJECUCIÓN  # OBTENER LISTA DE ARCHIVOS CSV
global_files = [f for f in files if GLOBAL_FILE_PATTERN in os.path.basename(f)]  # FILTRAR EL CSV GLOBAL  # FILTRAR ARCHIVOS GLOBALES
if not global_files:  # VERIFICAR SI NO HAY ARCHIVOS GLOBALES
    raise FileNotFoundError(f"[ ERROR ] No se encontró archivo con patrón '{GLOBAL_FILE_PATTERN}'")  # LANZAR ERROR SI NO ENCONTRADO

df_global = pd.read_csv(global_files[0])                             # LEER CSV GLOBAL  # CARGAR DATAFRAME GLOBAL
if 'anomaly' not in df_global.columns:  # VERIFICAR COLUMNA ANOMALY
    raise ValueError("[ ERROR ] No se encontró columna 'anomaly' en IF global")  # LANZAR ERROR SI FALTA COLUMNA
if 'sequence' not in df_global.columns:  # VERIFICAR COLUMNA SEQUENCE
    raise ValueError("[ ERROR ] No existe columna 'sequence' en IF global")  # LANZAR ERROR SI FALTA COLUMNA

def contar_secuencias(col):  # DEFINIR FUNCIÓN CONTAR SECUENCIAS
    seq = col.fillna(0).reset_index(drop=True)  # PREPARAR SERIE DE SECUENCIAS
    n = len(seq)  # OBTENER LONGITUD
    count = 0  # INICIALIZAR CONTADOR
    i = 0  # INICIALIZAR ÍNDICE
    while i < n:  # BUCLE PRINCIPAL
        if seq[i] == 0:  # SI ES CERO
            i += 1  # INCREMENTAR ÍNDICE
            continue  # CONTINUAR
        # Detectar bloque no-cero  # COMENTARIO DE DETECCIÓN
        j = i  # INICIAR J EN I
        bloque = []  # INICIALIZAR BLOQUE
        while j < n and seq[j] != 0:  # BUCLE PARA BLOQUE
            bloque.append(seq[j])  # AGREGAR A BLOQUE
            j += 1  # INCREMENTAR J
        # Contar si el bloque tiene al menos una secuencia creciente consecutiva  # COMENTARIO DE CONTEO
        if len(bloque) >= 2 and all(bloque[k] == bloque[k-1] + 1 for k in range(1, len(bloque))):  # VERIFICAR SECUENCIA CRECIENTE
            count += 1  # INCREMENTAR CONTADOR
        i = j  # saltar al siguiente bloque  # ACTUALIZAR I A J
    return count  # RETORNAR CONTADOR

# DEFINIR VARIABLES GLOBALES  
y_true_global = df_global['anomaly']                                  # ANOMALÍAS REALES GLOBALES  # OBTENER ANOMALÍAS REALES
total_global = int(y_true_global.sum())  # CALCULAR TOTAL GLOBAL
total_seq= contar_secuencias(df_global['sequence'])                                                           # TOTAL DE ANOMALÍAS  # CONTAR SECUENCIAS
max_sequence_global = int(df_global['sequence'].max())                # LONGITUD MÁXIMA DE SECUENCIA  # OBTENER MÁXIMA SECUENCIA
y_pred_global = y_true_global   # ASIGNAR PREDICCIÓN GLOBAL

# MÉTRICAS GLOBALES  
tp_global = ((y_true_global == 1) & (y_pred_global == 1)).sum()  # CALCULAR TP GLOBAL
fp_global = ((y_true_global == 0) & (y_pred_global == 1)).sum()  # CALCULAR FP GLOBAL
fn_global = ((y_true_global == 1) & (y_pred_global == 0)).sum()  # CALCULAR FN GLOBAL

# CARGAR DEFINICIÓN DE PARTICIONES  
with open(CLUSTERS_JSON, 'r', encoding='utf-8') as f:  # ABRIR JSON
    clusters_json = json.load(f)  # CARGAR JSON
if SHOW_INFO:  # SI MOSTRAR INFO
    print(f"[ INFO ] Clusters cargados desde '{CLUSTERS_JSON}'")  # IMPRIMIR INFO

# CARGAR DATAFRAMES DE PARTICIONES  
cluster_dfs = {}  # INICIALIZAR DICCIONARIO DFS
cluster_names = []  # INICIALIZAR LISTA NOMBRES
for cluster_name, subclusters in clusters_json.items():  # BUCLE POR CLUSTERS
    for sub_name in subclusters.keys():  # BUCLE POR SUBCLUSTERS
        file_name = f"cluster_{cluster_name}_{sub_name}.csv"  # DEFINIR NOMBRE ARCHIVO
        file_path = os.path.join(EXECUTION_FOLDER, file_name)  # DEFINIR RUTA ARCHIVO
        if os.path.exists(file_path):  # SI EXISTE ARCHIVO
            df_sub = pd.read_csv(file_path)  # CARGAR DATAFRAME SUB
            cluster_key = f"{cluster_name}_{sub_name}"  # DEFINIR CLAVE CLUSTER
            cluster_dfs[cluster_key] = df_sub  # AGREGAR A DICCIONARIO
            cluster_names.append(cluster_key)  # AGREGAR A LISTA NOMBRES
        else:  # SI NO EXISTE
            if SHOW_INFO:  # SI MOSTRAR INFO
                print(f"[ ADVERTENCIA ] No encontrado: {file_path}")  # IMPRIMIR ADVERTENCIA

# ÍNDICES DE ANOMALÍAS GLOBALES  
ad_indices = df_global[df_global['anomaly'] == 1].index.tolist()  # OBTENER ÍNDICES ANOMALÍAS

# CALCULAR RI E II + USAR datetime DEL GLOBAL  
intermediates = []  # INICIALIZAR LISTA INTERMEDIOS
for idx in ad_indices:  # BUCLE POR ÍNDICES
    ri = 0  # INICIALIZAR RI
    ii = 0.0  # INICIALIZAR II
    partition_scores = {}  # INICIALIZAR DICCIONARIO SCORES
    for c_name, df_c in cluster_dfs.items():  # BUCLE POR CLUSTERS DFS
        if idx < len(df_c) and df_c.loc[idx, 'anomaly'] == 1:  # SI ÍNDICE VÁLIDO Y ANOMALÍA
            score = df_c.loc[idx, 'anomaly_score']  # OBTENER SCORE
            ri += 1  # INCREMENTAR RI
            ii += score  # SUMAR A II
            partition_scores[c_name] = round(score, 4)  # AGREGAR SCORE REDONDEADO
        else:  # SI NO
            partition_scores[c_name] = 0.0  # ASIGNAR CERO

    # USAR datetime DEL GLOBAL (sin transformación)  # COMENTARIO DE DATETIME
    datetime_str = df_global.loc[idx, 'datetime']  # OBTENER DATETIME
    if pd.isna(datetime_str):  # SI ES NAN
        datetime_str = "NaT"  # ASIGNAR NAT
    else:  # SI NO
        # Si es timestamp, convertir a string legible  # COMENTARIO DE CONVERSIÓN
        if isinstance(datetime_str, pd.Timestamp):  # SI ES TIMESTAMP
            datetime_str = datetime_str.strftime('%Y-%m-%d %H:%M:%S')  # FORMATEAR STRING
        else:  # SI NO
            datetime_str = str(datetime_str)  # CONVERTIR A STRING

    intermediates.append({  # AGREGAR A INTERMEDIOS
        'global_index': int(idx),  # ÍNDICE GLOBAL
        'datetime': datetime_str,  # ← Valor directo del global  # DATETIME
        'global_anomaly_score': round(df_global.loc[idx, 'anomaly_score'], 4),  # SCORE GLOBAL REDONDEADO
        'ri': ri,  # RI
        'ii': round(ii, 4),  # II REDONDEADO
        **partition_scores  # SCORES PARTICIONES
    })

# CREAR DATAFRAME Y ORDENAR  
df_intermediates = pd.DataFrame(intermediates)  # CREAR DATAFRAME INTERMEDIOS
df_intermediates = df_intermediates.sort_values(by='ii', ascending=False)  # ORDENAR POR II DESCENDENTE

# REORDENAR COLUMNAS  
cols = ['datetime', 'global_index', 'global_anomaly_score', 'ri', 'ii'] + cluster_names  # DEFINIR COLUMNAS
df_intermediates = df_intermediates[cols]  # REORDENAR DATAFRAME

# GUARDAR ARCHIVO DE ESTADÍSTICAS  
df_intermediates.to_csv(INTERMEDIATES_CSV, index=False)  # GUARDAR CSV INTERMEDIOS
if SHOW_INFO:  # SI MOSTRAR INFO
    print(f"[ GUARDADO ] Archivo RI/II con datetime (del global) → '{INTERMEDIATES_CSV}'")  # IMPRIMIR GUARDADO

# === GENERAR CSV DE MÉTRICAS FINALES ===  
csv_rows = [{  # INICIAR LISTA FILAS CSV
    'file': 'global',  # ARCHIVO GLOBAL
    'anomalies_real': total_global,  # ANOMALÍAS REALES
    'anomalies_detected': total_global,  # ANOMALÍAS DETECTADAS
    'detections_correct': tp_global,  # DETECCIONES CORRECTAS
    'false_positives': fp_global,  # FALSOS POSITIVOS
    'false_negatives': fn_global,  # FALSOS NEGATIVOS
    'total_sequences': total_seq,  # TOTAL SECUENCIAS
    'max_sequence': max_sequence_global,  # MÁXIMA SECUENCIA
    'precision': round(precision_score(y_true_global, y_pred_global, zero_division=0), 4),  # CALCULAR PRECISIÓN
    'recall': round(recall_score(y_true_global, y_pred_global, zero_division=0), 4),  # CALCULAR RECALL
    'f1_score': round(f1_score(y_true_global, y_pred_global, zero_division=0), 4),  # CALCULAR F1
    'accuracy': round(accuracy_score(y_true_global, y_pred_global), 4),  # CALCULAR ACCURACY
    'mcc': round(matthews_corrcoef(y_true_global, y_pred_global), 4),  # CALCULAR MCC
    'ratio_detection': round(recall_score(y_true_global, y_pred_global, zero_division=0), 4),  # RATIO DETECCIÓN
    'ratio_fp': round(fp_global / len(y_true_global), 4) if len(y_true_global) > 0 else 0,  # RATIO FP
    'perc_global_anomalies_detected': 100.0,  # PORCENTAJE GLOBAL DETECTADO
    'perc_cluster_vs_global': 100.0,  # PORCENTAJE CLUSTER VS GLOBAL
    'total_coincidences': tp_global  # TOTAL COINCIDENCIAS
}]

# PROCESAR CADA PARTICIÓN  
for cluster_name, subclusters in clusters_json.items():  # BUCLE POR CLUSTERS
    for sub_name in subclusters.keys():  # BUCLE POR SUBCLUSTERS
        file_name = f"cluster_{cluster_name}_{sub_name}.csv"  # DEFINIR NOMBRE ARCHIVO
        file_path = os.path.join(EXECUTION_FOLDER, file_name)  # DEFINIR RUTA ARCHIVO
        cluster_key = f"{cluster_name}_{sub_name}"  # DEFINIR CLAVE CLUSTER
        if not os.path.exists(file_path):  # SI NO EXISTE ARCHIVO
            csv_rows.append({'file': cluster_key, 'error': 'archivo no encontrado'})  # AGREGAR ERROR
            continue  # CONTINUAR
        df_sub = pd.read_csv(file_path)  # CARGAR DATAFRAME SUB
        if 'anomaly' not in df_sub.columns or 'sequence' not in df_sub.columns:  # VERIFICAR COLUMNAS
            csv_rows.append({'file': cluster_key, 'error': 'falta anomaly o sequence'})  # AGREGAR ERROR
            continue  # CONTINUAR

        total_seq = int(df_sub['sequence'].max()) if df_sub['sequence'].notna().any() else 0  # CALCULAR TOTAL SEQ
        y_pred = df_sub['anomaly']  # OBTENER PREDICCIÓN
        tp = ((y_true_global == 1) & (y_pred == 1)).sum()  # CALCULAR TP
        fp = ((y_true_global == 0) & (y_pred == 1)).sum()  # CALCULAR FP
        fn = ((y_true_global == 1) & (y_pred == 0)).sum()  # CALCULAR FN
        total_cluster_anomalies = int(y_pred.sum())  # TOTAL ANOMALÍAS CLUSTER

        perc_global = round(tp / total_global * 100, 2) if total_global > 0 else 0  # PORCENTAJE GLOBAL
        perc_cluster = round(tp / total_cluster_anomalies * 100, 2) if total_cluster_anomalies > 0 else 0  # PORCENTAJE CLUSTER

        csv_rows.append({  # AGREGAR FILA CSV
            'file': cluster_key,  # ARCHIVO
            'anomalies_real': total_global,  # ANOMALÍAS REALES
            'anomalies_detected': total_cluster_anomalies,  # ANOMALÍAS DETECTADAS
            'detections_correct': int(tp),  # DETECCIONES CORRECTAS
            'false_positives': int(fp),  # FALSOS POSITIVOS
            'false_negatives': int(fn),  # FALSOS NEGATIVOS
            'total_sequences': total_seq,  # TOTAL SECUENCIAS
            'max_sequence': total_seq,  # MÁXIMA SECUENCIA
            'precision': round(precision_score(y_true_global, y_pred, zero_division=0), 4),  # PRECISIÓN
            'recall': round(recall_score(y_true_global, y_pred, zero_division=0), 4),  # RECALL
            'f1_score': round(f1_score(y_true_global, y_pred, zero_division=0), 4),  # F1
            'accuracy': round(accuracy_score(y_true_global, y_pred), 4),  # ACCURACY
            'mcc': round(matthews_corrcoef(y_true_global, y_pred), 4),  # MCC
            'ratio_detection': round(recall_score(y_true_global, y_pred, zero_division=0), 4),  # RATIO DETECCIÓN
            'ratio_fp': round(fp / len(y_true_global), 4) if len(y_true_global) > 0 else 0,  # RATIO FP
            'perc_global_anomalies_detected': perc_global,  # PORCENTAJE GLOBAL DETECTADO
            'perc_cluster_vs_global': perc_cluster,  # PORCENTAJE CLUSTER VS GLOBAL
            'total_coincidences': int(tp)  # TOTAL COINCIDENCIAS
        })

# GUARDAR CSV FINAL  
df_csv = pd.DataFrame(csv_rows)[columns_order]  # CREAR DATAFRAME CSV
df_csv.to_csv(OUTPUT_CSV, index=False)  # GUARDAR CSV
if SHOW_INFO:  # SI MOSTRAR INFO
    print(f"[ GUARDADO ] CSV resumen de métricas en '{OUTPUT_CSV}'")  # IMPRIMIR GUARDADO