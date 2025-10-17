import pandas as pd  # MANEJO DE DATAFRAMES
import os            # RUTAS Y DIRECTORIOS
import json          # PARA LEER EL JSON

# CARGAR DATASET
df = pd.read_csv('../../results/preparation/05_variance.csv')
print("[ INFO ] Dataset cargado")

# CREAR CARPETA DE RESULTADOS
os.makedirs('../../results/execution', exist_ok=True)
print("[ INFO ] Carpeta '../../results/execution' creada si no existía")

# CARGAR DEFINICIÓN DE CLUSTERS DESDE JSON
json_path = 'clusters.json'
with open(json_path, 'r', encoding='utf-8') as f:
    clusters = json.load(f)
print(f"[ INFO ] Clusters cargados desde '{json_path}'")

# GUARDAR ARCHIVOS POR SUBPARTICIÓN
for cluster_name, subparts in clusters.items():
    for sub_name, cols in subparts.items():
        # FILTRAR COLUMNAS EXISTENTES
        cols_to_save = [c for c in cols if c in df.columns]
        if 'cluster_manual' in df.columns:
            cols_to_save.append('cluster_manual')  # MANTENER COLUMNA MANUAL
        cluster_df = df[cols_to_save]
        
        # GUARDAR CSV
        out_file = f"../../results/execution/cluster_{cluster_name}_{sub_name}.csv"
        cluster_df.to_csv(out_file, index=False)
        print(f"[ GUARDADO ] {out_file}")
