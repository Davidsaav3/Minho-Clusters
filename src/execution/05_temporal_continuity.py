import pandas as pd                              # IMPORTAR PANDAS PARA MANEJO DE DATAFRAMES
import glob                                      # IMPORTAR GLOB PARA BUSCAR ARCHIVOS POR PATRÓN
import os                                        # IMPORTAR OS PARA MANEJO DE RUTAS Y CARPETAS

# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
RESULTS_FOLDER = '../../results'                 # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # SUBCARPETA DE EJECUCIÓN
GLOBAL_FILE = os.path.join(EXECUTION_FOLDER, '04_global.csv') # CSV GLOBAL DE IF
CLUSTER_PATTERN = os.path.join(EXECUTION_FOLDER, 'cluster_*.csv') # PATRÓN PARA LOS CSV DE CLUSTERS
SHOW_INFO = True                                 # MOSTRAR INFORMACIÓN EN CONSOLA

# BUSCAR TODOS LOS ARCHIVOS CLUSTER QUE NO TERMINEN EN '_if.csv'
cluster_files = [
    f for f in glob.glob(CLUSTER_PATTERN)
    if not f.endswith('_if.csv')
]

# FUNCIÓN PARA CALCULAR SECUENCIAS DE ANOMALÍAS
def add_sequence_column(df, anomaly_col, output_col):
    """
    Calcula las secuencias consecutivas de valores 1 en la columna de anomalías indicada
    y añade una nueva columna con los tamaños de cada secuencia.

    Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        anomaly_col (str): Nombre de la columna de anomalías (por ejemplo 'anomaly' o 'genuine_anomaly').
        output_col (str): Nombre de la nueva columna de secuencias (por ejemplo 'sequence' o 'genuine_sequence').

    Devuelve:
        (total_seq, max_seq): Número total de secuencias y longitud máxima detectada.
    """
    if anomaly_col not in df.columns:
        df[output_col] = 0
        return 0, 0

    vals = df[anomaly_col].values                # OBTENER VALORES DE LA COLUMNA DE ANOMALÍAS
    seq = [0] * len(vals)                        # INICIALIZAR LISTA DE LONGITUDES DE SECUENCIAS
    current = 0                                  # CONTADOR DE SECUENCIA ACTUAL
    total_seq = 0                                # CONTADOR DE SECUENCIAS TOTALES
    max_seq = 0                                  # LONGITUD MÁXIMA DE SECUENCIA

    # RECORRER TODA LA COLUMNA DE ANOMALÍAS
    for i, v in enumerate(vals):
        if v == 1:                               # SI ES ANOMALÍA
            current += 1
            seq[i] = current                     # GUARDAR LONGITUD ACTUAL
            if current == 1:
                total_seq += 1                   # NUEVA SECUENCIA DETECTADA
            if current > max_seq:
                max_seq = current
        else:
            current = 0                          # REINICIAR SI NO ES ANOMALÍA

    df[output_col] = seq                         # AÑADIR COLUMNA DE SECUENCIAS AL DATAFRAME
    return total_seq, max_seq                    # DEVOLVER RESULTADOS


# PROCESAR TODOS LOS ARCHIVOS (GLOBAL + CLUSTERS)
files_to_process = [GLOBAL_FILE] + cluster_files

for file_path in files_to_process:
    if not os.path.exists(file_path):
        print(f"[ SKIP ] Archivo no encontrado: {file_path}")
        continue

    df = pd.read_csv(file_path)                  # CARGAR CSV
    filename = os.path.basename(file_path)

    # SECUENCIAS DE ANOMALÍAS REALES (COLUMNA 'anomaly')
    if 'anomaly' in df.columns:
        total_seq, max_seq = add_sequence_column(df, 'anomaly', 'sequence')
        if SHOW_INFO:
            print(f"[ OK ] {filename}: {total_seq} secuencias reales detectadas (longitud máx = {max_seq})")
    else:
        if SHOW_INFO:
            print(f"[ SKIP ] {filename}: no contiene columna 'anomaly' -> no se crea 'sequence'.")

    # SECUENCIAS DE ANOMALÍAS GENUINAS (COLUMNA 'genuine_anomaly')
    # Solo se calcula para el archivo global
    if file_path == GLOBAL_FILE and 'genuine_anomaly' in df.columns:
        total_gseq, max_gseq = add_sequence_column(df, 'genuine_anomaly', 'genuine_sequence')
        if SHOW_INFO:
            print(f"[ OK ] {filename}: {total_gseq} secuencias genuinas detectadas (longitud máx = {max_gseq})")
    else:
        if SHOW_INFO and file_path != GLOBAL_FILE:
            print(f"[ SKIP ] {filename}: se omite genuine_anomaly en clusters.")
        elif SHOW_INFO:
            print(f"[ SKIP ] {filename}: no contiene columna 'genuine_anomaly' -> no se crea 'genuine_sequence'.")

    # GUARDAR CSV ACTUALIZADO
    df.to_csv(file_path, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] {filename} actualizado correctamente.\n")
