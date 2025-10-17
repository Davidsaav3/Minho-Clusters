# src/04_if_por_cluster.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import glob
import os

os.makedirs('../results', exist_ok=True)

# Archivos de clusters a procesar
files = glob.glob('../../results/execution/cluster-*.csv')

for file_path in files:
    print(f"\nProcesando archivo: {file_path}")
    df = pd.read_csv(file_path)

    # Detectar columnas numéricas para IF
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        print(f"No hay columnas numéricas en {file_path}, se salta.")
        continue

    # Aplicar Isolation Forest
    print(f"Aplicando Isolation Forest a {len(df)} filas y {len(num_cols)} columnas...")
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    df['anomaly'] = clf.fit_predict(df[num_cols])
    # Mapear: 1 = normal, -1 = anómalo → 0 = normal, 1 = anómalo
    df['anomaly'] = df['anomaly'].map({1:0, -1:1})

    # Guardar resultado con sufijo _if
    df.to_csv(file_path, index=False)
    print(f"Clustering por filas aplicado y sobrescrito en {file_path}")
