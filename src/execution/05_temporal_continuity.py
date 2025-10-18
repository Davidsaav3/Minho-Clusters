import pandas as pd                              # IMPORTAR PANDAS PARA MANEJO DE DATAFRAMES
import glob                                      # IMPORTAR GLOB PARA BUSCAR ARCHIVOS POR PATRÓN
import os                                        # IMPORTAR OS PARA MANEJO DE RUTAS Y CARPETAS

# PARÁMETROS DE CONFIGURACIÓN
RESULTS_FOLDER = '../../results'                # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN
IF_GLOBAL_FILE = os.path.join(EXECUTION_FOLDER, 'if_global.csv')  # CSV GLOBAL DE IF
INPUT_IF_CSV = os.path.join(EXECUTION_FOLDER, '01_if.csv')        # CSV DE IF POR CLUSTERS
CLUSTER_PATTERN = os.path.join(EXECUTION_FOLDER, 'cluster_*.csv') # PATRÓN DE CSV DE CLUSTERS
SHOW_INFO = True                                # MOSTRAR INFORMACIÓN EN CONSOLA

# FUNCIÓN AUXILIAR PARA CALCULAR SECUENCIAS DE ANOMALÍAS
def add_sequence_column(df, anomaly_col):
    vals = df[anomaly_col].values                # OBTENER VALORES DE LA COLUMNA DE ANOMALÍAS
    seq = [0]*len(vals)                          # INICIALIZAR LISTA PARA LONGITUD DE SECUENCIAS
    current = 0                                  # CONTADOR DE SECUENCIA ACTUAL
    total_seq = 0                                # CONTADOR DE SECUENCIAS TOTALES
    max_seq = 0                                  # LONGITUD MÁXIMA DE SECUENCIA
    for i, v in enumerate(vals):                # RECORRER TODOS LOS VALORES
        if v == 1:                               # SI ES ANOMALÍA
            current += 1                          # INCREMENTAR SECUENCIA ACTUAL
            seq[i] = current                      # GUARDAR LONGITUD ACTUAL EN LISTA
            if current == 1:
                total_seq += 1                    # NUEVA SECUENCIA DETECTADA
            if current > max_seq:
                max_seq = current                 # ACTUALIZAR LONGITUD MÁXIMA
        else:
            current = 0                            # REINICIAR SECUENCIA SI NO HAY ANOMALÍA
    df['sequence'] = seq                          # AÑADIR COLUMNA DE SECUENCIA AL DATAFRAME
    return total_seq, max_seq                     # DEVOLVER TOTAL DE SECUENCIAS Y LONGITUD MÁXIMA

# PROCESAR IF_GLOBAL.CSV
if os.path.exists(IF_GLOBAL_FILE):
    df_global = pd.read_csv(IF_GLOBAL_FILE)      # CARGAR CSV GLOBAL
    is_anomaly_col = df_global['is_anomaly'] if 'is_anomaly' in df_global.columns else pd.Series([0]*len(df_global), name='is_anomaly')  # OBTENER COLUMNA is_anomaly
    total_seq, max_seq = add_sequence_column(df_global, 'anomaly')  # CALCULAR SECUENCIAS DE ANOMALÍAS
    df_global['is_anomaly'] = is_anomaly_col     # MANTENER COLUMNA ORIGINAL is_anomaly
    df_global.to_csv(IF_GLOBAL_FILE, index=False) # GUARDAR CSV ACTUALIZADO
    if SHOW_INFO:
        print(f"[ GUARDADO ] if_global.csv actualizado con secuencias")
        print(f"[ INFO ] Secuencias globales: {total_seq}, longitud máxima: {max_seq}")

# PROCESAR 01_IF.CSV
df_if = pd.read_csv(INPUT_IF_CSV)               # CARGAR CSV DE IF POR CLUSTERS
is_anomaly_col = df_if['is_anomaly'] if 'is_anomaly' in df_if.columns else pd.Series([0]*len(df_if), name='is_anomaly')  # OBTENER COLUMNA is_anomaly
total_seq, max_seq = add_sequence_column(df_if, 'anomaly')  # CALCULAR SECUENCIAS
df_if['is_anomaly'] = is_anomaly_col            # MANTENER COLUMNA ORIGINAL is_anomaly
df_if.to_csv(INPUT_IF_CSV, index=False)         # GUARDAR CSV ACTUALIZADO
if SHOW_INFO:
    print(f"[ GUARDADO ] 01_if.csv actualizado con secuencias")
    print(f"[ INFO ] Secuencias IF: {total_seq}, longitud máxima: {max_seq}")

# PROCESAR CSV DE CLUSTERS
cluster_files = glob.glob(CLUSTER_PATTERN)      # LISTAR TODOS LOS CSV DE CLUSTERS
for file_path in cluster_files:
    df_cluster = pd.read_csv(file_path)         # CARGAR CSV DEL CLUSTER
    if 'anomaly' not in df_cluster.columns:     # OMITIR SI NO HAY COLUMNA ANOMALY
        if SHOW_INFO:
            print(f"[ SKIP ] No hay columna 'anomaly' en {file_path}")
        continue

    is_anomaly_col = df_cluster['is_anomaly'] if 'is_anomaly' in df_cluster.columns else pd.Series([0]*len(df_cluster), name='is_anomaly')  # OBTENER COLUMNA is_anomaly
    total_seq, max_seq = add_sequence_column(df_cluster, 'anomaly')  # CALCULAR SECUENCIAS
    df_cluster['is_anomaly'] = is_anomaly_col    # MANTENER COLUMNA ORIGINAL is_anomaly
    df_cluster.to_csv(file_path, index=False)    # GUARDAR CSV ACTUALIZADO
    if SHOW_INFO:
        print(f"[ GUARDADO ] {file_path} actualizado con secuencias")
        print(f"[ INFO ] Secuencias: {total_seq}, longitud máxima: {max_seq}")
