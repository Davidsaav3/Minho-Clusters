import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA
import glob
import os

# PARÁMETROS CONFIGURABLES
RESULTS_FOLDER = '../../results'                         # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DONDE SE GUARDAN RESULTADOS
CLUSTER_PATTERN = 'cluster_*.csv'                       # PATRÓN PARA ARCHIVOS DE CLUSTERS A PROCESAR

USE_PCA = True                # APLICAR REDUCCIÓN DIMENSIONAL CON PCA SI ES TRUE
N_PCA_COMPONENTS = 20         # NÚMERO MÁXIMO DE COMPONENTES PCA A CONSERVAR

CLUSTER_METHOD = 'minibatch'  # MÉTODO DE CLUSTERING: 'kmeans', 'minibatch', 'dbscan', 'birch'
N_CLUSTERS = 4                # NÚMERO DE CLUSTERS PARA KMEANS, MINIBATCH Y BIRCH
EPS_DBSCAN = 0.5              # DISTANCIA MÁXIMA ENTRE PUNTOS PARA DBSCAN
MIN_SAMPLES_DBSCAN = 5        # NÚMERO MÍNIMO DE PUNTOS EN RADIO EPS PARA DBSCAN
SHOW_INFO = True              # MOSTRAR INFORMACIÓN EN PANTALLA

# CREAR CARPETAS DE RESULTADOS SI NO EXISTEN
os.makedirs(RESULTS_FOLDER, exist_ok=True)           # CREAR CARPETA PRINCIPAL
os.makedirs(EXECUTION_FOLDER, exist_ok=True)         # CREAR CARPETA DE EJECUCIÓN
if SHOW_INFO:
    print(f"[ INFO ] CARPETAS CREADAS SI NO EXISTÍAN")

# LISTAR ARCHIVOS QUE CUMPLEN EL PATRÓN
files = glob.glob(os.path.join(EXECUTION_FOLDER, CLUSTER_PATTERN))
if SHOW_INFO:
    print(f"[ INFO ] ARCHIVOS ENCONTRADOS PARA CLUSTERING: {len(files)}")

# PROCESAR CADA ARCHIVO DE CLUSTERS
for file_path in files:
    df = pd.read_csv(file_path)                      # CARGAR DATAFRAME
    if SHOW_INFO:
        print(f"[ INFO ] PROCESANDO {file_path}, DIMENSIONES: {df.shape}")

    # SELECCIONAR SOLO COLUMNAS NUMÉRICAS PARA CLUSTERING
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        if SHOW_INFO:
            print("[ SKIP ] NO HAY COLUMNAS NUMÉRICAS, OMITIENDO ARCHIVO")
        continue
    if SHOW_INFO:
        print(f"[ INFO ] COLUMNAS NUMÉRICAS DISPONIBLES: {len(num_cols)}")

    # REDUCCIÓN DIMENSIONAL OPCIONAL CON PCA
    if USE_PCA:
        n_components = min(N_PCA_COMPONENTS, len(num_cols))  # AJUSTAR COMPONENTES SI HAY MENOS COLUMNAS
        pca = PCA(n_components=n_components, random_state=42) # CREAR OBJETO PCA
        X_input = pca.fit_transform(df[num_cols])            # TRANSFORMAR DATOS
        if SHOW_INFO:
            print(f"[ INFO ] PCA APLICADO, COMPONENTES: {X_input.shape[1]}")
    else:
        X_input = df[num_cols].values                         # USAR TODAS LAS COLUMNAS NUMÉRICAS
        if SHOW_INFO:
            print("[ INFO ] USANDO TODAS LAS COLUMNAS NUMÉRICAS SIN PCA")

    # SELECCIÓN DEL MÉTODO DE CLUSTERING
    if CLUSTER_METHOD == 'minibatch':
        model = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=1000, random_state=42)
    elif CLUSTER_METHOD == 'birch':
        model = Birch(n_clusters=N_CLUSTERS)
    elif CLUSTER_METHOD == 'dbscan':
        model = DBSCAN(eps=EPS_DBSCAN, min_samples=MIN_SAMPLES_DBSCAN)
    elif CLUSTER_METHOD == 'kmeans':
        model = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    else:
        raise ValueError(f"[ ERROR ] MÉTODO DE CLUSTERING DESCONOCIDO: {CLUSTER_METHOD}")

    # AJUSTAR MODELO Y OBTENER PREDICCIONES DE CLUSTERS
    df['cluster'] = model.fit_predict(X_input)                # PREDICCIÓN DE CLUSTERS
    n_clusters_found = len(set(df['cluster']))                # CONTAR NÚMERO DE CLUSTERS ENCONTRADOS
    if SHOW_INFO:
        print(f"[ RESULTADO ] CLUSTERS ENCONTRADOS: {n_clusters_found}")

    # GUARDAR DATAFRAME CON CLUSTERS EN EL MISMO ARCHIVO
    df.to_csv(file_path, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] CLUSTERING APLICADO EN {file_path}")
