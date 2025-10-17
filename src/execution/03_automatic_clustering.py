import pandas as pd
from sklearn.cluster import MiniBatchKMeans  # CLUSTERING
from sklearn.decomposition import PCA  # REDUCCIÓN DIMENSIONAL
import glob
import os

# CREAR CARPETA DE RESULTADOS
os.makedirs('../../results', exist_ok=True)

# ARCHIVOS A PROCESAR
files = glob.glob('../../results/execution/cluster-*.csv')

# PROCESAR CADA ARCHIVO
for file_path in files:
    df = pd.read_csv(file_path)
    print(f"[ INFO ] Dimensiones: {df.shape}")
    
    # COLUMNAS NUMÉRICAS PARA CLUSTERING
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"[ INFO ] Columnas numéricas: {len(num_cols)}")
    if len(num_cols) == 0:
        print("[ SKIP ] No hay columnas numéricas")
        continue
    
    # REDUCCIÓN DIMENSIONAL CON PCA (YA ESCALADO ANTERIORMENTE)
    n_components = min(20, len(num_cols))
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(df[num_cols])
    
    # MINI BATCH KMEANS
    print("[ CLUSTERING ] MiniBatchKMeans")
    mbk = MiniBatchKMeans(n_clusters=4, batch_size=1000, random_state=42)
    df['cluster'] = mbk.fit_predict(X_reduced)
    n_clusters = len(set(df['cluster']))
    print(f"[ RESULTADO ] Clusters encontrados: {n_clusters}")
    
    # GUARDAR RESULTADO
    df.to_csv(file_path, index=False)
    print(f"[ GUARDADO ] Clustering aplicado en {file_path}")
