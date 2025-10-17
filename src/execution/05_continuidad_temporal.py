# src/05_continuidad_temporal.py
import pandas as pd
import glob
import os

os.makedirs('../results', exist_ok=True)

# Archivos a procesar: IF global + IF por clusters
files = glob.glob('../../results/execution/cluster-*.csv')

for file_path in files:
    print(f"\nProcesando archivo: {file_path}")
    df = pd.read_csv(file_path)

    if 'anomaly' not in df.columns and 'anomaly_global' not in df.columns:
        print(f"No se encuentra columna de anomalías en {file_path}, se salta.")
        continue

    # Determinar qué columna usar
    anomaly_col = 'anomaly' if 'anomaly' in df.columns else 'anomaly_global'

    # Calcular longitud de secuencias consecutivas de anomalías
    seq_len = []
    count = 0
    for val in df[anomaly_col]:
        if val == 1:
            count += 1
        else:
            count = 0
        seq_len.append(count)

    df['sequence'] = seq_len

    # Guardar resultado con sufijo _seq
    df.to_csv(file_path, index=False)
    print(f"Clustering por filas aplicado y sobrescrito en {file_path}")        
