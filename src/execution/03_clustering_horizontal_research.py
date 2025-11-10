import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, Birch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import glob
import os
import itertools
import numpy as np

# PARÁMETROS 
RESULTS_FOLDER = '../../results'                         # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DONDE SE GUARDAN RESULTADOS
CLUSTER_PATTERN = 'cluster_*.csv'                       # PATRÓN PARA ARCHIVOS DE CLUSTERS A PROCESAR
INPUT_CSV = os.path.join(EXECUTION_FOLDER, '01_contaminated.csv')  # ARCHIVO DE ENTRADA
OUTPUT_CSV = os.path.join(EXECUTION_FOLDER, '03_global.csv')  # SALIDA CSV BASE

USE_PCA = False                # INDICA SI SE APLICA PCA PARA REDUCIR DIMENSIONALIDAD
N_PCA_COMPONENTS = 20          # NÚMERO MÁXIMO DE COMPONENTES PRINCIPALES A CONSERVAR

CLUSTER_METHODS = ['kmeans', 'minibatch', 'birch', 'dbscan']  # LISTA DE ALGORITMOS A PROBAR
# 
N_CLUSTERS_LIST = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]               # LISTA DE N_CLUSTERS PARA KMEANS, MINI-BATCH Y BIRCH
EPS_DBSCAN_LIST = [1.5, 2, 5, 10, 20]         # LISTA DE EPS A PROBAR EN DBSCAN
MIN_SAMPLES_DBSCAN_LIST = [2, 4, 6, 8, 10, 15, 20, 25, 30, 35]  # LISTA DE MIN_SAMPLES A PROBAR EN DBSCAN
BATCH_SIZE_LIST = [1000, 2000, 3000, 4000]           # LISTA DE TAMAÑOS DE LOTE PARA MINI-BATCH Y BIRCH

SHOW_INFO = True               # SI TRUE, MUESTRA INFORMACIÓN DETALLADA EN CONSOLA

# FUNCION PARA APLICAR CLUSTERING A UN DATAFRAME
def apply_clustering(df, method='kmeans', n_clusters=4, eps=5.0, min_samples=80, batch_size=10000, show_info=True):
    # SELECCIONAR SOLO COLUMNAS NUMÉRICAS PARA CLUSTERING
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        if show_info:
            print("[ SKIP ] NO HAY COLUMNAS NUMÉRICAS, OMITIENDO DATAFRAME")
        return df, None, None

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
    labels = model.fit_predict(X_input)
    df['cluster'] = labels
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    if show_info:
        print(f"[ RESULTADO ] CLUSTERS ENCONTRADOS: {n_clusters_found}")

    return df, X_input, labels

# PROCESAR EL ARCHIVO GLOBAL DE ENTRADA
df_global = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] PROCESANDO ARCHIVO GLOBAL: {INPUT_CSV}, DIMENSIONES: {df_global.shape}")

# CREAR LA VERSIÓN BASE 03_global CON KMEANS
df_base, _, _ = apply_clustering(df_global.copy(), method='kmeans', n_clusters=4, show_info=SHOW_INFO)
df_base.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] ARCHIVO GLOBAL BASE CON CLUSTERS GUARDADO EN: {OUTPUT_CSV}")

# CREAR VARIACIONES PRINCIPALES CON LÍMITE DE 100 EJECUCIONES
max_iterations = 100
iteration_count = 0

# ALMACENAR SCORES PARA SELECCIONAR MEJOR CONFIGURACIÓN
all_results = []

for method in CLUSTER_METHODS:
    # DEFINIR RANGOS DE PARÁMETROS SEGÚN MÉTODO
    if method in ['kmeans', 'minibatch', 'birch']:
        param_combinations = list(itertools.product(N_CLUSTERS_LIST, BATCH_SIZE_LIST))
    elif method == 'dbscan':
        param_combinations = list(itertools.product(EPS_DBSCAN_LIST, MIN_SAMPLES_DBSCAN_LIST))
    else:
        param_combinations = [(4,)]  # FALLBACK

    for params in param_combinations:
        # SALIR SI SE ALCANZA EL LÍMITE
        if iteration_count >= max_iterations:
            break

        df_copy = df_global.copy()

        # APLICAR CLUSTERING SEGÚN MÉTODO
        if method in ['kmeans', 'minibatch', 'birch']:
            n_clusters_val, batch_val = params
            df_result, X_input, labels = apply_clustering(df_copy, method=method, n_clusters=n_clusters_val, batch_size=batch_val, show_info=SHOW_INFO)
            param_str = f"n{n_clusters_val}_batch{batch_val}"
        elif method == 'dbscan':
            eps_val, min_samples_val = params
            df_result, X_input, labels = apply_clustering(df_copy, method=method, eps=eps_val, min_samples=min_samples_val, show_info=SHOW_INFO)
            param_str = f"eps{eps_val}_min{min_samples_val}"

        # ---------------- CRITERIO DE EVALUACIÓN: MÉTRICAS DE CLUSTERING ----------------
        score_silhouette = -1    # INICIALIZAR SILHOUETTE SCORE
        score_calinski = -1      # INICIALIZAR CALINSKI-HARABASZ SCORE
        score_davies = np.inf    # INICIALIZAR DAVIES-BOULDIN SCORE

        # CALCULAR METRICAS SOLO SI HAY MÁS DE 1 CLUSTER Y MENOS DE N-1 CLUSTERS
        if labels is not None and len(set(labels)) > 1 and len(set(labels)) < len(labels):
            try:
                score_silhouette = silhouette_score(X_input, labels)
                score_calinski = calinski_harabasz_score(X_input, labels)
                score_davies = davies_bouldin_score(X_input, labels)
            except:
                pass

        # GUARDAR RESULTADOS EN LISTA
        all_results.append({
            'method': method,
            'params': param_str,
            'silhouette': score_silhouette,
            'calinski': score_calinski,
            'davies': score_davies
        })

        # GUARDAR CSV DE ESTA CONFIGURACIÓN
        filename = f"03_global_{method}_{param_str}.csv"
        output_path = os.path.join(EXECUTION_FOLDER, filename)
        df_result.to_csv(output_path, index=False)
        if SHOW_INFO:
            print(f"[ GUARDADO ] VARIACIÓN {filename} GUARDADA")
            print(f"[ SCORE ] Silhouette: {score_silhouette:.4f}, Calinski: {score_calinski:.4f}, Davies: {score_davies:.4f}")

        iteration_count += 1  # INCREMENTAR CONTADOR DE ITERACIONES

    if iteration_count >= max_iterations:
        break

# ---------------- SELECCIONAR MEJOR CONFIGURACIÓN SEGÚN SILHOUETTE SCORE ----------------
best_config = max(all_results, key=lambda x: x['silhouette'])
print("\n[ MEJOR CONFIGURACIÓN SEGÚN SILHOUETTE SCORE ]")
print(f"MÉTODO: {best_config['method']}")
print(f"PARÁMETROS: {best_config['params']}")
print(f"SILHOUETTE SCORE: {best_config['silhouette']:.4f}")
print(f"CALINSKI-HARABASZ: {best_config['calinski']:.4f}")
print(f"DAVIES-BOULDIN: {best_config['davies']:.4f}")
