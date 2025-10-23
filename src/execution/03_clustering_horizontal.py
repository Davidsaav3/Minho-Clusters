import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA
import glob
import os

# PARÁMETROS 
RESULTS_FOLDER = '../../results'                         # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DONDE SE GUARDAN RESULTADOS
CLUSTER_PATTERN = 'cluster_*.csv'                       # PATRÓN PARA ARCHIVOS DE CLUSTERS A PROCESAR
INPUT_CSV = os.path.join(EXECUTION_FOLDER, '01_contaminated.csv')
OUTPUT_CSV = os.path.join(EXECUTION_FOLDER, '03_global.csv')  # SALIDA IF GLOBAL

USE_PCA = False                
# SI ES TRUE, SE APLICA PCA (ANÁLISIS DE COMPONENTES PRINCIPALES) PARA REDUCIR LA DIMENSIONALIDAD DEL DATASET.
# PCA TRANSFORMA LAS VARIABLES ORIGINALES EN NUEVAS VARIABLES ORTOGONALES (COMPONENTES PRINCIPALES) 
# QUE RETIENEN LA MÁXIMA VARIANZA POSIBLE. ESTO AYUDA A MEJORAR RENDIMIENTO Y VISUALIZACIÓN.

N_PCA_COMPONENTS = 20         
# NÚMERO MÁXIMO DE COMPONENTES PRINCIPALES A CONSERVAR.
# - MÁXIMO: NO PUEDE SUPERAR EL NÚMERO DE VARIABLES ORIGINALES DEL DATASET.
# - MÍNIMO: 1 (AL MENOS UNA COMPONENTE PRINCIPAL PARA REPRESENTAR VARIANZA).
# - NORMAL: SE ELIGEN COMPONENTES SUFICIENTES PARA RETENER ENTRE 80%-95% DE LA VARIANZA TOTAL.
# LIMITA LA DIMENSIONALIDAD DEL DATASET Y RETIENE SOLO LA VARIANZA MÁS RELEVANTE.

# PARÁMETROS DE CLUSTERING
CLUSTER_METHOD = 'dbscan'  
# MÉTODO DE CLUSTERING A UTILIZAR:
# 'kmeans'     -> CLUSTERING CLÁSICO, AGRUPA EN N_CLUSTERS BASADO EN DISTANCIA
# 'minibatch'  -> K-MEANS POR LOTES, MÁS RÁPIDO PARA GRANDES DATASETS
# 'dbscan'     -> CLUSTERING BASADO EN DENSIDAD, NO REQUIERE N_CLUSTERS, DETECTA RUIDO
# 'birch'      -> CLUSTERING HIERÁRQUICO PARA GRANDES DATASETS, CONSTRUIR ÁRBOLES DE CLUSTERS

N_CLUSTERS = 4                
# NÚMERO DE CLUSTERS A GENERAR PARA MÉTODOS QUE LO REQUIEREN (KMEANS, MINIBATCH, BIRCH)

EPS_DBSCAN = 1.5              
# RADIO MÁXIMO PARA CONSIDERAR PUNTOS COMO VECINOS EN DBSCAN. 
# DOS PUNTOS DENTRO DE EPS PUEDEN PERTENECER AL MISMO CLUSTER.

MIN_SAMPLES_DBSCAN = 15
# NÚMERO MÍNIMO DE PUNTOS NECESARIOS DENTRO DEL RADIO EPS PARA FORMAR UN CLUSTER EN DBSCAN.
# PUNTOS CON MENOS VECINOS SE CONSIDERAN RUIDO.

SHOW_INFO = True              
# SI ES TRUE, MUESTRA INFORMACIÓN DETALLADA DEL PROCESO EN CONSOLA.

# -------------------------
# FUNCION PARA APLICAR CLUSTERING A UN DATAFRAME
# -------------------------
def apply_clustering(df, show_info=True):
    # SELECCIONAR SOLO COLUMNAS NUMÉRICAS PARA CLUSTERING
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        if show_info:
            print("[ SKIP ] NO HAY COLUMNAS NUMÉRICAS, OMITIENDO DATAFRAME")
        return df

    if show_info:
        print(f"[ INFO ] COLUMNAS NUMÉRICAS DISPONIBLES: {len(num_cols)}")

    # REDUCCIÓN DIMENSIONAL OPCIONAL CON PCA
    if USE_PCA:
        n_components = min(N_PCA_COMPONENTS, len(num_cols), len(df))  
        pca = PCA(n_components=n_components, random_state=42)
        X_input = pca.fit_transform(df[num_cols])
        if show_info:
            print(f"[ INFO ] PCA APLICADO, COMPONENTES: {X_input.shape[1]}")
    else:
        X_input = df[num_cols].values
        if show_info:
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
    df['cluster'] = model.fit_predict(X_input)
    n_clusters_found = len(set(df['cluster']))
    if show_info:
        print(f"[ RESULTADO ] CLUSTERS ENCONTRADOS: {n_clusters_found}")

    return df

# -------------------------
# PROCESAR ARCHIVOS DE CLUSTERS
# -------------------------
files = glob.glob(os.path.join(EXECUTION_FOLDER, CLUSTER_PATTERN))
if SHOW_INFO:
    print(f"[ INFO ] ARCHIVOS ENCONTRADOS PARA CLUSTERING: {len(files)}")

for file_path in files:
    df = pd.read_csv(file_path)
    if SHOW_INFO:
        print(f"[ INFO ] PROCESANDO {file_path}, DIMENSIONES: {df.shape}")

    df = apply_clustering(df, show_info=SHOW_INFO)

    df.to_csv(file_path, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] CLUSTERING APLICADO EN {file_path}")

# -------------------------
# PROCESAR EL ARCHIVO GLOBAL DE ENTRADA
# -------------------------
df_global = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] PROCESANDO ARCHIVO GLOBAL: {INPUT_CSV}, DIMENSIONES: {df_global.shape}")

df_global = apply_clustering(df_global, show_info=SHOW_INFO)

df_global.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] ARCHIVO GLOBAL CON CLUSTERS GUARDADO EN: {OUTPUT_CSV}")
