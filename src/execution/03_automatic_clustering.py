import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA
import glob
import os

# CONFIGURACIÓN
CLUSTER_METHOD = 'minibatch'  # OPCIONES: 'kmeans', 'minibatch', 'agglomerative', 'dbscan', 'birch'
N_CLUSTERS = 4                # Para métodos que lo necesitan (KMeans, MiniBatchKMeans, Agglomerative, Birch)
EPS_DBSCAN = 0.5              # Para DBSCAN
MIN_SAMPLES_DBSCAN = 5        # Para DBSCAN
USE_PCA = True                # True = aplicar reducción de dimensionalidad con PCA, False = usar todas las columnas numéricas
N_PCA_COMPONENTS = 20         # Número máximo de componentes PCA

# CREAR CARPETA DE RESULTADOS
os.makedirs('../../results', exist_ok=True)

# ARCHIVOS A PROCESAR
files = glob.glob('../../results/execution/cluster_*.csv')

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
    
    # REDUCCIÓN DIMENSIONAL OPCIONAL CON PCA
    if USE_PCA:
        n_components = min(N_PCA_COMPONENTS, len(num_cols))
        pca = PCA(n_components=n_components, random_state=42)
        X_input = pca.fit_transform(df[num_cols])
        print(f"[ INFO ] Reducción dimensional aplicada: {X_input.shape[1]} componentes")
    else:
        X_input = df[num_cols].values
        print(f"[ INFO ] Usando todas las columnas numéricas sin PCA")
    
    # SELECCIÓN DE MÉTODO DE CLUSTERING
    if CLUSTER_METHOD == 'kmeans':
        model = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    elif CLUSTER_METHOD == 'minibatch':
        model = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=1000, random_state=42)
    elif CLUSTER_METHOD == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    elif CLUSTER_METHOD == 'dbscan':
        model = DBSCAN(eps=EPS_DBSCAN, min_samples=MIN_SAMPLES_DBSCAN)
    elif CLUSTER_METHOD == 'birch':
        model = Birch(n_clusters=N_CLUSTERS)
    else:
        raise ValueError(f"[ ERROR ] Método de clustering desconocido: {CLUSTER_METHOD}")
    
    # AJUSTE DEL MODELO Y PREDICCIÓN
    df['cluster'] = model.fit_predict(X_input)
    n_clusters_found = len(set(df['cluster']))
    print(f"[ RESULTADO ] Clusters encontrados: {n_clusters_found}")
    
    # GUARDAR RESULTADO
    df.to_csv(file_path, index=False)
    print(f"[ GUARDADO ] Clustering aplicado en {file_path}")
