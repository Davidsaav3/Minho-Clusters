import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, Birch
from sklearn.decomposition import PCA
import os

# PARÁMETROS GENERALES
RESULTS_FOLDER = '../../results'  # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN

INPUT_CSV = os.path.join(EXECUTION_FOLDER, '01_contaminated.csv')  # CSV DE ENTRADA
OUTPUT_CSV_BASE = os.path.join(EXECUTION_FOLDER, '03_global_predictive')  # BASE NOMBRE CSV DE SALIDA

USE_PCA = False                # SI TRUE, SE APLICA PCA
N_PCA_COMPONENTS = 20          # NÚMERO DE COMPONENTES PRINCIPALES SI PCA
CLUSTER_METHOD = 'kmeans'      # MÉTODO DE CLUSTERING ('kmeans', 'minibatch', 'birch', 'dbscan')
N_CLUSTERS = 4                 # NÚMERO DE CLUSTERS PARA KMEANS/MINIBATCH/BIRCH
EPS_DBSCAN = 5.0               # RADIO EPS PARA DBSCAN
MIN_SAMPLES_DBSCAN = 80        # MÍNIMO DE MUESTRAS PARA DBSCAN
BATCH_SIZE = 10000             # TAMAÑO DE LOTE PARA MINIBATCH KMEANS
SHOW_INFO = True               # SI TRUE MUESTRA INFO EN CONSOLA
CHUNK_SIZE = 25                # NÚMERO DE FILAS POR BLOQUE EN EL FLUJO

# FUNCIÓN DE CLUSTERING
# RETORNA DATAFRAME CON COLUMNA 'cluster'
def apply_clustering(df, show_info=True):
    # OBTENER COLUMNAS NUMÉRICAS PARA CLUSTERING (EXCLUIR 'is_anomaly')
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'is_anomaly' in num_cols:
        num_cols.remove('is_anomaly')

    # SI NO HAY COLUMNAS NUMÉRICAS, SALIR
    if len(num_cols) == 0:
        print("[SKIP] NO HAY COLUMNAS NUMÉRICAS PARA CLUSTERING")
        return df, None

    # IMPUTAR NAN CON LA MEDIA DE CADA COLUMNA
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # APLICAR PCA SI ESTÁ ACTIVADO
    if USE_PCA:
        n_components = min(N_PCA_COMPONENTS, len(num_cols), len(df))
        pca = PCA(n_components=n_components, random_state=42)
        X_input = pca.fit_transform(df[num_cols])
        if show_info:
            print(f"[INFO] PCA APLICADO CON {n_components} COMPONENTES")
    else:
        X_input = df[num_cols].values
        if show_info:
            print("[INFO] USANDO TODAS LAS COLUMNAS NUMÉRICAS SIN PCA")

    # SELECCIONAR MODELO DE CLUSTERING SEGÚN PARÁMETRO
    if CLUSTER_METHOD == 'minibatch':
        model = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=BATCH_SIZE, random_state=42)
    elif CLUSTER_METHOD == 'birch':
        model = Birch(n_clusters=N_CLUSTERS)
    elif CLUSTER_METHOD == 'dbscan':
        model = DBSCAN(eps=EPS_DBSCAN, min_samples=MIN_SAMPLES_DBSCAN)
    elif CLUSTER_METHOD == 'kmeans':
        model = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    else:
        raise ValueError(f"[ERROR] MÉTODO DE CLUSTERING DESCONOCIDO: {CLUSTER_METHOD}")

    # ELIMINAR COLUMNA 'cluster' SI YA EXISTE PARA REENTRENAR
    if 'cluster' in df.columns:
        df = df.drop(columns=['cluster'])

    # AJUSTAR MODELO Y CREAR COLUMNA 'cluster'
    df = pd.concat([df.reset_index(drop=True),
                    pd.Series(model.fit_predict(X_input), name='cluster')],
                   axis=1)

    # ASEGURAR TIPO ENTERO EN LA COLUMNA 'cluster'
    df['cluster'] = df['cluster'].astype(int)

    if show_info:
        print(f"[RESULTADO] CLUSTERS ENCONTRADOS: {len(set(df['cluster']))}")

    return df, model

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV DE ENTRADA
if SHOW_INFO:
    print(f"[INFO] ARCHIVO CARGADO: {INPUT_CSV} ({df.shape[0]} FILAS)")

# DIVIDIR DATASET 50/50 
split_index = int(len(df) * 0.5)
train_df = df.iloc[:split_index].copy()   # PRIMER 50% PARA ENTRENAMIENTO
stream_df = df.iloc[split_index:].copy()  # SEGUNDO 50% PARA FLUJO INCREMENTAL
if SHOW_INFO:
    print(f"[INFO] DIVISIÓN 50/50 ORDENADA -> ENTRENAMIENTO: {train_df.shape[0]} | FLUJO: {stream_df.shape[0]}")

# CLUSTERING INICIAL
train_df, model = apply_clustering(train_df, show_info=SHOW_INFO)  # CLUSTERING SOBRE TRAIN

# IDENTIFICAR CLUSTERS CON MÁS ANOMALÍAS
if 'is_anomaly' in train_df.columns:
    cluster_counts = train_df.groupby('cluster')['is_anomaly'].sum().sort_values(ascending=False)
    top_clusters = cluster_counts.index[:max(1, len(cluster_counts)//2)].tolist()
    if SHOW_INFO:
        print(f"[INFO] CLUSTERS CON ANOMALÍAS RELEVANTES: {top_clusters}")
else:
    top_clusters = list(range(N_CLUSTERS))
    print("[WARN] NO SE ENCONTRÓ COLUMNA 'is_anomaly', SE USARÁN TODOS LOS CLUSTERS")

# COLUMNAS NUMÉRICAS PARA IMPUTACIÓN EN FLUJO
num_cols = [c for c in train_df.select_dtypes(include=['int64', 'float64']).columns if c != 'is_anomaly']
col_means = train_df[num_cols].mean()

# FLUJO INCREMENTAL
file_counter = 1  # CONTADOR DE ARCHIVOS SALIDA

for start in range(0, len(stream_df), CHUNK_SIZE):
    chunk = stream_df.iloc[start:start+CHUNK_SIZE].copy()  # EXTRAER BLOQUE DE FLUJO

    # IMPUTAR NAN SOLO EN COLUMNAS EXISTENTES DEL BLOQUE
    existing_num_cols = [c for c in num_cols if c in chunk.columns]
    chunk[existing_num_cols] = chunk[existing_num_cols].fillna(col_means[existing_num_cols])

    # ACTUALIZAR TRAINING: ELIMINAR FILAS ANTIGUAS Y AÑADIR BLOQUE NUEVO
    train_df = pd.concat([train_df.iloc[CHUNK_SIZE:], chunk], ignore_index=True)

    # RECALCULAR CLUSTERS CON EL NUEVO TRAINING
    train_df, model = apply_clustering(train_df, show_info=False)

    # IDENTIFICAR NUEVOS CLUSTERS CON MÁS ANOMALÍAS
    if 'is_anomaly' in train_df.columns:
        cluster_counts = train_df.groupby('cluster')['is_anomaly'].sum().sort_values(ascending=False)
        top_clusters = cluster_counts.index[:max(1, len(cluster_counts)//2)].tolist()
    else:
        top_clusters = list(range(N_CLUSTERS))

    # AÑADIR COLUMNA 'predictive' CON CLUSTERS SEPARADOS POR GUION MEDIO
    train_df['predictive'] = '-'.join([str(c) for c in top_clusters]) if top_clusters else ''

    # ACTUALIZAR MEDIA DE COLUMNAS NUMÉRICAS PARA SIGUIENTE BLOQUE
    col_means = train_df[num_cols].mean()

    # GUARDAR BLOQUE ACTUAL EN CSV
    output_csv = f"{OUTPUT_CSV_BASE}_{file_counter}.csv"
    train_df.to_csv(output_csv, index=False)

    if SHOW_INFO:
        print(f"[FLUJO] BLOQUE {file_counter} CLUSTERS CON ANOMALÍAS RELEVANTES {top_clusters} -> {output_csv} | Predictive: {train_df['predictive'].iloc[0]}")

    file_counter += 1

if SHOW_INFO:
    print("[FIN] FLUJO INCREMENTAL COMPLETADO")