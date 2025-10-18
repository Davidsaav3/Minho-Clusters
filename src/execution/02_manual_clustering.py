import pandas as pd  # MANEJO DE DATAFRAMES
import os            # RUTAS Y DIRECTORIOS
import json          # PARA LEER EL JSON

# PARÁMETROS CONFIGURABLES
RESULTS_FOLDER = '../../results'                       # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN
INPUT_CSV = '../../results/preparation/00_contaminated.csv'       # DATASET DE ENTRADA
CLUSTERS_JSON = 'clusters.json'                      # ARCHIVO JSON CON DEFINICIÓN DE CLUSTERS
MANUAL_COLUMN = 'cluster_manual'                     # NOMBRE DE LA COLUMNA MANUAL (SI EXISTE)
SHOW_INFO = True                                     # MOSTRAR INFO EN PANTALLA

# CREAR CARPETAS DE RESULTADOS
os.makedirs(RESULTS_FOLDER, exist_ok=True)           # CREAR CARPETA PRINCIPAL SI NO EXISTE
os.makedirs(EXECUTION_FOLDER, exist_ok=True)        # CREAR CARPETA DE EJECUCIÓN SI NO EXISTE
if SHOW_INFO:
    print("[ INFO ] Carpetas creadas si no existían")

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)                          # CARGAR CSV DE VARIANZA
if SHOW_INFO:
    print(f"[ INFO ] Dataset cargado desde '{INPUT_CSV}'")

# CARGAR DEFINICIÓN DE CLUSTERS DESDE JSON
with open(CLUSTERS_JSON, 'r', encoding='utf-8') as f:
    clusters = json.load(f)                          # CARGAR CONFIGURACIÓN DE CLUSTERS
if SHOW_INFO:
    print(f"[ INFO ] Clusters cargados desde '{CLUSTERS_JSON}'")

# GUARDAR ARCHIVOS POR SUBPARTICIÓN
for cluster_name, subparts in clusters.items():
    for sub_name, cols in subparts.items():
        # FILTRAR COLUMNAS EXISTENTES EN EL DATAFRAME
        cols_to_save = [c for c in cols if c in df.columns]
        if MANUAL_COLUMN in df.columns:
            cols_to_save.append(MANUAL_COLUMN)       # MANTENER COLUMNA MANUAL SI EXISTE
        
        # CREAR DATAFRAME PARA EL SUBCLUSTER
        cluster_df = df[cols_to_save]
        
        # GUARDAR CSV DEL SUBCLUSTER
        out_file = os.path.join(EXECUTION_FOLDER, f"cluster_{cluster_name}_{sub_name}.csv")
        cluster_df.to_csv(out_file, index=False)
        if SHOW_INFO:
            print(f"[ GUARDADO ] {out_file}")
