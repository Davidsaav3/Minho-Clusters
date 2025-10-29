import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, Birch
from sklearn.decomposition import PCA
import glob
import os
import itertools

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
# AHORA SE HARÁN VARIACIONES CON CADA UNO DE LOS 4 ALGORITMOS
CLUSTER_METHODS = ['kmeans', 'minibatch', 'dbscan', 'birch']

#---------------------------------------------------------------------------------
N_CLUSTERS_LIST = [3, 4, 5]              
# LISTA DE N_CLUSTERS A PROBAR PARA KMEANS, MINI-BATCH Y BIRCH

EPS_DBSCAN_LIST = [4.0, 5.0, 6.0]              
# LISTA DE EPS A PROBAR EN DBSCAN

MIN_SAMPLES_DBSCAN_LIST = [50, 80, 100]        
# LISTA DE MIN_SAMPLES A PROBAR EN DBSCAN

BATCH_SIZE_LIST = [5000, 10000]                
# LISTA DE TAMAÑOS DE LOTE PARA MINI-BATCH KMEANS Y BIRCH

SHOW_INFO = True              
# SI ES TRUE, MUESTRA INFORMACIÓN DETALLADA DEL PROCESO EN CONSOLA.

# FUNCION PARA APLICAR CLUSTERING A UN DATAFRAME
def apply_clustering(df, method='kmeans', n_clusters=4, eps=5.0, min_samples=80, batch_size=10000, show_info=True):
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
    if method == 'minibatch':
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
    elif method == 'birch':
        model = Birch(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError(f"[ ERROR ] MÉTODO DE CLUSTERING DESCONOCIDO: {method}")

    # AJUSTAR MODELO Y OBTENER PREDICCIONES DE CLUSTERS
    df['cluster'] = model.fit_predict(X_input)
    n_clusters_found = len(set(df['cluster']))
    if show_info:
        print(f"[ RESULTADO ] CLUSTERS ENCONTRADOS: {n_clusters_found}")
    
    return df

# PROCESAR EL ARCHIVO GLOBAL DE ENTRADA
df_global = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] PROCESANDO ARCHIVO GLOBAL: {INPUT_CSV}, DIMENSIONES: {df_global.shape}")

# CREAR LA VERSIÓN BASE 03_global
df_base = apply_clustering(df_global.copy(), method='kmeans', n_clusters=4, show_info=SHOW_INFO)
df_base.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] ARCHIVO GLOBAL BASE CON CLUSTERS GUARDADO EN: {OUTPUT_CSV}")


# CREAR VARIACIONES PRINCIPALES
for method in CLUSTER_METHODS:
    # DEFINIR RANGOS DE PARÁMETROS SEGÚN MÉTODO
    if method in ['kmeans', 'minibatch', 'birch']:
        param_combinations = list(itertools.product(N_CLUSTERS_LIST, BATCH_SIZE_LIST))
    elif method == 'dbscan':
        param_combinations = list(itertools.product(EPS_DBSCAN_LIST, MIN_SAMPLES_DBSCAN_LIST))
    else:
        param_combinations = [(4,)]  # fallback

    for params in param_combinations:
        df_copy = df_global.copy()
        if method in ['kmeans', 'minibatch', 'birch']:
            n_clusters_val, batch_val = params
            df_result = apply_clustering(df_copy, method=method, n_clusters=n_clusters_val, batch_size=batch_val, show_info=SHOW_INFO)
            param_str = f"n{n_clusters_val}_batch{batch_val}"
        elif method == 'dbscan':
            eps_val, min_samples_val = params
            df_result = apply_clustering(df_copy, method=method, eps=eps_val, min_samples=min_samples_val, show_info=SHOW_INFO)
            param_str = f"eps{eps_val}_min{min_samples_val}"

        # GUARDAR CSV CON NOMBRE QUE REFLEJE MÉTODO Y PARÁMETROS
        filename = f"03_global_{method}_{param_str}.csv"
        output_path = os.path.join(EXECUTION_FOLDER, filename)
        df_result.to_csv(output_path, index=False)
        if SHOW_INFO:
            print(f"[ GUARDADO ] VARIACIÓN {filename} GUARDADA")
