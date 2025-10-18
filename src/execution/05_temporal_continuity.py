import pandas as pd  # MANEJO DE DATAFRAMES
import glob          # BUSCAR ARCHIVOS POR PATRÓN
import os            # MANEJO DE RUTAS Y CREACIÓN DE CARPETAS

# PARÁMETROS CONFIGURABLES
RESULTS_FOLDER = '../../results'                      # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN
IF_GLOBAL_FILE = os.path.join(EXECUTION_FOLDER, 'if_global.csv')  # ARCHIVO IF GLOBAL
CLUSTER_PATTERN = 'cluster_*.csv'                    # PATRÓN PARA ARCHIVOS DE CLUSTER
ANOMALY_GLOBAL_COLS = ['anomaly_global', 'anomaly']  # COLUMNAS DE ANOMALÍAS GLOBALES POSIBLES
ANOMALY_CLUSTER_COLS = ['anomaly', 'anomaly_global'] # COLUMNAS DE ANOMALÍAS DE CLUSTER POSIBLES
SHOW_INFO = True                                     # TRUE = MOSTRAR INFO EN PANTALLA

# CREAR CARPETAS NECESARIAS
os.makedirs(RESULTS_FOLDER, exist_ok=True)          # CREAR CARPETA PRINCIPAL SI NO EXISTE
os.makedirs(EXECUTION_FOLDER, exist_ok=True)        # CREAR CARPETA EJECUCIÓN SI NO EXISTE
if SHOW_INFO:
    print(f"[ INFO ] Carpetas creadas si no existían")

# FUNCIÓN AUXILIAR
def add_sequence_column(df, anomaly_col):
    """
    AÑADE COLUMNA 'sequence' CON LONGITUD DE SECUENCIA DE 1s
    DEVUELVE TOTAL DE SECUENCIAS Y LONGITUD MÁXIMA
    """
    vals = df[anomaly_col].values       # EXTRAER VALORES DE LA COLUMNA
    seq = [0]*len(vals)                 # INICIALIZAR SECUENCIA
    current = 0                          # CONTADOR ACTUAL
    total_seq = 0                        # CONTADOR TOTAL DE SECUENCIAS
    max_seq = 0                          # LONGITUD MÁXIMA DE SECUENCIA
    for i, v in enumerate(vals):
        if v == 1:
            current += 1
            seq[i] = current
            if current == 1:
                total_seq += 1          # NUEVA SECUENCIA DETECTADA
            if current > max_seq:
                max_seq = current       # ACTUALIZAR MÁXIMO
        else:
            current = 0                  # REINICIAR CONTADOR AL ENCONTRAR 0
    df['sequence'] = seq                 # AÑADIR COLUMNA 'sequence' AL DATAFRAME
    return total_seq, max_seq            # RETORNAR TOTAL Y MÁXIMO

# PROCESAR IF GLOBAL
if os.path.exists(IF_GLOBAL_FILE):
    df_global = pd.read_csv(IF_GLOBAL_FILE)                       # CARGAR CSV GLOBAL
    
    # DETECTAR COLUMNA DE ANOMALÍAS
    anomaly_col = next((c for c in ANOMALY_GLOBAL_COLS if c in df_global.columns), None)
    if anomaly_col:
        total_seq, max_seq = add_sequence_column(df_global, anomaly_col)  # CALCULAR SECUENCIAS
        df_global.to_csv(IF_GLOBAL_FILE, index=False)                     # GUARDAR CSV MODIFICADO
        if SHOW_INFO:
            print(f"[ GUARDADO ] Secuencias añadidas en '{IF_GLOBAL_FILE}'")
            print(f"[ INFO ] Secuencias globales: {total_seq}, longitud máxima: {max_seq}")
    else:
        if SHOW_INFO:
            print("[ SKIP ] No se encontró columna de anomalías en IF global")
else:
    if SHOW_INFO:
        print(f"[ SKIP ] Archivo '{IF_GLOBAL_FILE}' no encontrado")

# PROCESAR ARCHIVOS DE CLUSTERS
cluster_files = glob.glob(os.path.join(EXECUTION_FOLDER, CLUSTER_PATTERN))  # LISTAR ARCHIVOS DE CLUSTER

for file_path in cluster_files:
    df = pd.read_csv(file_path)                                  # CARGAR CSV DE CLUSTER
    
    # DETECTAR COLUMNA DE ANOMALÍAS
    anomaly_col = next((c for c in ANOMALY_CLUSTER_COLS if c in df.columns), None)
    if anomaly_col:
        total_seq, max_seq = add_sequence_column(df, anomaly_col)  # CALCULAR SECUENCIAS
        df.to_csv(file_path, index=False)                            # GUARDAR CSV MODIFICADO
        if SHOW_INFO:
            print(f"[ GUARDADO ] Secuencias añadidas en {file_path}")
            print(f"[ INFO ] Secuencias: {total_seq}, longitud máxima: {max_seq}")
    else:
        if SHOW_INFO:
            print(f"[ SKIP ] No hay columna de anomalías en {file_path}")
