# src/03_cluster_filas.py
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob
import os

os.makedirs('../results', exist_ok=True)

# Archivos a procesar: IF global + todos los cluster_*.csv
files = glob.glob('../../results/execution/cluster-*.csv')

for file_path in files:
    print(f"\n=== Procesando {file_path} ===")
    df = pd.read_csv(file_path)
    print(f"Dimensiones: {df.shape}")
    
    # Seleccionar solo columnas numéricas para clustering
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Número de columnas numéricas: {len(num_cols)}")
    if len(num_cols) == 0:
        print("No hay columnas numéricas para clustering, se salta.")
        continue
    
    # Escalado
    print("Escalando datos...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols])
    
    # Reducir dimensionalidad con PCA
    n_components = min(20, X_scaled.shape[1])
    print(f"Reduciendo dimensionalidad a {n_components} componentes con PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    
    # ======================
    # MiniBatchKMeans (reemplaza DBSCAN)
    # ======================
    print("Aplicando MiniBatchKMeans...")
    mbk = MiniBatchKMeans(n_clusters=4, batch_size=1000, random_state=42)
    df['cluster'] = mbk.fit_predict(X_reduced)
    n_clusters = len(set(df['cluster']))
    print(f"MiniBatchKMeans: {n_clusters} clusters encontrados")
    
    # Guardar resultado con sufijo _fila_cluster
    df.to_csv(file_path, index=False)
    print(f"Clustering por filas aplicado y sobrescrito en {file_path}")    
