import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA
import glob
import os

# PARÁMETROS CONFIGURABLES
RESULTS_FOLDER = '../../results'                     # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN
CLUSTER_PATTERN = 'cluster_*.csv'                   # PATRÓN PARA ARCHIVOS DE CLUSTERS

USE_PCA = True                # TRUE = APLICAR PCA, FALSE = USAR TODAS LAS COLUMNAS NUMÉRICAS
N_PCA_COMPONENTS = 20         # NÚMERO MÁXIMO DE COMPONENTES PCA A CONSERVAR

CLUSTER_METHOD = 'minibatch'  # MÉTODO DE CLUSTERING: 'kmeans', 'minibatch', 'dbscan', 'birch'
N_CLUSTERS = 4                # NÚMERO DE CLUSTERS (KMEANS, MINIBATCH, BIRCH)
EPS_DBSCAN = 0.5              # DISTANCIA MÁXIMA ENTRE PUNTOS PARA DBSCAN
MIN_SAMPLES_DBSCAN = 5        # MÍNIMO DE PUNTOS EN RADIO EPS PARA DBSCAN
SHOW_INFO = True              # MOSTRAR INFO EN PANTALLA

# CREAR CARPETAS DE RESULTADOS
os.makedirs(RESULTS_FOLDER, exist_ok=True)         # CREAR CARPETA PRINCIPAL SI NO EXISTE
os.makedirs(EXECUTION_FOLDER, exist_ok=True)      # CREAR CARPETA DE EJECUCIÓN SI NO EXISTE
if SHOW_INFO:
    print(f"[ INFO ] Carpeta '{RESULTS_FOLDER}' creada si no existía")

# LISTAR ARCHIVOS DE CLUSTERS
files = glob.glob(os.path.join(EXECUTION_FOLDER, CLUSTER_PATTERN))

# PROCESAR CADA ARCHIVO
for file_path in files:
    df = pd.read_csv(file_path)
    if SHOW_INFO:
        print(f"[ INFO ] Dimensiones: {df.shape}")
    
    # COLUMNAS NUMÉRICAS PARA CLUSTERING
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        if SHOW_INFO:
            print("[ SKIP ] No hay columnas numéricas")
        continue
    if SHOW_INFO:
        print(f"[ INFO ] Columnas numéricas: {len(num_cols)}")
    
    # REDUCCIÓN DIMENSIONAL OPCIONAL CON PCA
    if USE_PCA:
        n_components = min(N_PCA_COMPONENTS, len(num_cols))
        pca = PCA(n_components=n_components, random_state=42)
        X_input = pca.fit_transform(df[num_cols])
        if SHOW_INFO:
            print(f"[ INFO ] Reducción dimensional aplicada: {X_input.shape[1]} componentes")
    else:
        X_input = df[num_cols].values
        if SHOW_INFO:
            print(f"[ INFO ] Usando todas las columnas numéricas sin PCA")
    
    # SELECCIÓN DE MÉTODO DE CLUSTERING
    if CLUSTER_METHOD == 'minibatch':
        model = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=1000, random_state=42)
    elif CLUSTER_METHOD == 'birch':
        model = Birch(n_clusters=N_CLUSTERS)
    elif CLUSTER_METHOD == 'dbscan':
        model = DBSCAN(eps=EPS_DBSCAN, min_samples=MIN_SAMPLES_DBSCAN)
    elif CLUSTER_METHOD == 'kmeans':
        model = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    else:
        raise ValueError(f"[ ERROR ] Método de clustering desconocido: {CLUSTER_METHOD}")
    
    # AJUSTE DEL MODELO Y PREDICCIÓN
    df['cluster'] = model.fit_predict(X_input)
    n_clusters_found = len(set(df['cluster']))
    if SHOW_INFO:
        print(f"[ RESULTADO ] Clusters encontrados: {n_clusters_found}")
    
    # GUARDAR RESULTADO
    df.to_csv(file_path, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] Clustering aplicado en {file_path}")
