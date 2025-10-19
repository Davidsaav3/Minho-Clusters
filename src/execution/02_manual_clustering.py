import pandas as pd
import os
import json

# PARÁMETROS
RESULTS_FOLDER = '../../results'                         # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN
INPUT_CSV = '../../results/execution/00_contaminated.csv'     # CSV DE ENTRADA
CLUSTERS_JSON = 'clusters.json'                         # JSON CON DEFINICIÓN DE CLUSTERS
MANUAL_COLUMN = 'cluster_manual'                        # COLUMNA ADICIONAL A MANTENER SI EXISTE
SHOW_INFO = True                                        # MOSTRAR INFORMACIÓN EN CONSOLA

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO DESDE '{INPUT_CSV}'")

# SEPARAR COLUMNA is_anomaly PARA GENERAR CLUSTERS
if 'is_anomaly' in df.columns:
    is_anomaly_column = df['is_anomaly']           # GUARDAR ETIQUETA ORIGINAL
    df_input = df.drop(columns=['is_anomaly'])     # ELIMINAR PARA NO USARLA EN CLUSTERS
else:
    df_input = df.copy()                           # COPIA COMPLETA SI NO EXISTE ETIQUETA
    is_anomaly_column = pd.Series([0]*len(df_input), name='is_anomaly')  # COLUMNA DE CEROS

# CARGAR DEFINICIÓN DE CLUSTERS DESDE JSON
with open(CLUSTERS_JSON, 'r', encoding='utf-8') as f:
    clusters = json.load(f)
if SHOW_INFO:
    print(f"[ INFO ] CLUSTERS CARGADOS DESDE '{CLUSTERS_JSON}'")

# CREAR CSV POR SUBCLUSTER
for cluster_name, subparts in clusters.items():
    for sub_name, cols in subparts.items():
        # FILTRAR COLUMNAS EXISTENTES EN EL DATASET
        cols_to_save = [c for c in cols if c in df_input.columns]
        
        # AÑADIR COLUMNA SI EXISTE
        if MANUAL_COLUMN in df_input.columns:
            cols_to_save.append(MANUAL_COLUMN)
        
        # CREAR DATAFRAME CON COLUMNAS SELECCIONADAS
        cluster_df = df_input[cols_to_save].copy()
        
        # AÑADIR COLUMNA is_anomaly AL FINAL PARA ANÁLISIS
        cluster_df['is_anomaly'] = is_anomaly_column  # RECUPERAR VALORES ORIGINALES
        
        # GUARDAR CSV DEL SUBCLUSTER
        out_file = os.path.join(EXECUTION_FOLDER, f"cluster_{cluster_name}_{sub_name}.csv")
        cluster_df.to_csv(out_file, index=False)
        if SHOW_INFO:
            print(f"[ GUARDADO ] {out_file}")
