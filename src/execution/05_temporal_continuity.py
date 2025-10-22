import pandas as pd                              # IMPORTAR PANDAS PARA MANEJO DE DATAFRAMES
import glob                                      # IMPORTAR GLOB PARA BUSCAR ARCHIVOS POR PATRÓN
import os                                        # IMPORTAR OS PARA MANEJO DE RUTAS Y CARPETAS

# PARÁMETROS DE CONFIGURACIÓN
RESULTS_FOLDER = '../../results'                # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN
GLOBAL_FILE = os.path.join(EXECUTION_FOLDER, '04_global.csv')  # CSV GLOBAL DE IF
CLUSTER_PATTERN = os.path.join(EXECUTION_FOLDER, 'cluster_*.csv') # PATRÓN DE CSV DE CLUSTERS
SHOW_INFO = True                                # MOSTRAR INFORMACIÓN EN CONSOLA

# BUSCAR ARCHIVOS QUE EMPIECEN POR 'cluster_' PERO NO TERMINEN EN '_if.csv'
cluster_files = [
    f for f in glob.glob(CLUSTER_PATTERN)
    if not f.endswith('_if.csv')
]

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

# LISTA DE TODOS LOS ARCHIVOS A PROCESAR
files_to_process = [GLOBAL_FILE] + cluster_files
# SE INCLUYEN ARCHIVOS GLOBALES Y TODOS LOS CSV DE CLUSTERS

# PROCESAR TODOS LOS ARCHIVOS
for file_path in files_to_process:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)             # CARGAR CSV
        if 'anomaly' not in df.columns:         # OMITIR SI NO HAY COLUMNA ANOMALY
            if SHOW_INFO:
                print(f"[ SKIP ] No hay columna 'anomaly' en {file_path}")
            continue

        total_seq, max_seq = add_sequence_column(df, 'anomaly')  # CALCULAR SECUENCIAS
        # LA COLUMNA 'is_anomaly' SE MANTIENE IGUAL

        df.to_csv(file_path, index=False)       # GUARDAR CSV ACTUALIZADO

        if SHOW_INFO:
            print(f"[ GUARDADO ] {os.path.basename(file_path)} actualizado con secuencias")
            print(f"[ INFO ] Secuencias: {total_seq}, longitud máxima: {max_seq}")
