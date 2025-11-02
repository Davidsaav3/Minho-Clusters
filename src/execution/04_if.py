import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import glob
import os
import numpy as np

# PARÁMETROS GENERALES
RESULTS_FOLDER = '../../results'  # Carpeta principal de resultados
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # Carpeta de ejecución
CLUSTER_PATTERN = 'cluster_*.csv'  # Patrón de archivos de clusters
CONTAMINATION_PATTERN = 'contamination_*.csv'  # Patrón de archivos contaminados

INPUT_CSV = '../../results/execution/03_global.csv'  # ARCHIVO GLOBAL DE ENTRADA
OUTPUT_CSV = os.path.join(EXECUTION_FOLDER, '04_global.csv')  # SALIDA IF GLOBAL
OUTPUT_IF_CSV = os.path.join(EXECUTION_FOLDER, '04_global_if.csv')  # SALIDA IF GLOBAL SOLO ANOMALÍAS
INPUT_UNCONTAMINATED_CSV = '../../results/preparation/05_variance_recortado.csv'  # NUEVO ARCHIVO IF SIN CONTAMINACIÓN
OUTPUT_UNCONTAMINATED_IF_CSV = os.path.join(EXECUTION_FOLDER, '04_global_if_uncontaminated.csv')  # SALIDA IF SIN CONTAMINACIÓN

SAVE_ANOMALY_CSV = True  # Guardar solo anomalías detectadas
SORT_ANOMALY_SCORE = True  # Ordenar CSV de anomalías por score
INCLUDE_SCORE = True  # Incluir columna 'anomaly_score' en CSV de anomalías
NORMALIZE_SCORE = True  # Normalizar anomaly_score entre 0 y 1
SHOW_INFO = True  # Mostrar información en consola

# HIPERPARÁMETROS ISOLATION FOREST
N_ESTIMATORS = 100
MAX_SAMPLES = 'auto'
CONTAMINATION = 0.01
MAX_FEATURES = 1.0
BOOTSTRAP = False
N_JOBS = -1
RANDOM_STATE = 42
VERBOSE = 0

# -----------------------------
# ISOLATION FOREST GLOBAL
# -----------------------------
if os.path.exists(INPUT_CSV):
    df = pd.read_csv(INPUT_CSV)
    if SHOW_INFO:
        print(f"[ INFO ] DATASET GLOBAL CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

    # Separar columnas de referencia
    if 'is_anomaly' in df.columns:
        is_anomaly_column = df['is_anomaly']
        cluster_column = df['cluster'] if 'cluster' in df.columns else pd.Series([None]*len(df), name='cluster')
        df_input = df.drop(columns=[col for col in ['is_anomaly', 'cluster'] if col in df.columns])
    else:
        is_anomaly_column = pd.Series([0]*len(df), name='is_anomaly')
        cluster_column = df['cluster'] if 'cluster' in df.columns else pd.Series([None]*len(df), name='cluster')
        df_input = df.copy()

    num_cols = df_input.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_input[num_cols])

    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=VERBOSE
    )
    clf.fit(df_scaled)

    anomaly_score = clf.decision_function(df_scaled) * -1
    pred = clf.predict(df_scaled)
    df['anomaly'] = np.where(pred == 1, 0, 1)
    df['anomaly_score'] = anomaly_score
    df['is_anomaly'] = is_anomaly_column
    df['cluster'] = cluster_column

    num_anomalies = df['anomaly'].sum()
    num_normals = df.shape[0] - num_anomalies
    if SHOW_INFO:
        print(f"[ INFO ] REGISTROS TOTALES (GLOBAL): {df.shape[0]}")
        print(f"[ INFO ] ANOMALÍAS DETECTADAS (GLOBAL): {num_anomalies}")
        print(f"[ INFO ] REGISTROS NORMALES (GLOBAL): {num_normals}")
        print(f"[ INFO ] PORCENTAJE ANOMALÍAS (GLOBAL): {num_anomalies/df.shape[0]*100:.2f}%")

    df.to_csv(OUTPUT_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] CSV COMPLETO CON ANOMALÍAS EN '{OUTPUT_CSV}'")

    if SAVE_ANOMALY_CSV:
        df_anomalies = df.loc[df['anomaly'] == 1].copy()
        df_anomalies['anomaly_score'] = df_anomalies['anomaly_score'].astype(float)
        if NORMALIZE_SCORE:
            min_score = df_anomalies['anomaly_score'].min()
            max_score = df_anomalies['anomaly_score'].max()
            if max_score > min_score:
                df_anomalies['anomaly_score'] = (df_anomalies['anomaly_score'] - min_score) / (max_score - min_score)
        if SORT_ANOMALY_SCORE:
            df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)
        if not INCLUDE_SCORE:
            df_anomalies.drop(columns=['anomaly_score'], inplace=True)
        df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
        if SHOW_INFO:
            print(f"[ GUARDADO ] CSV ANOMALÍAS {'ORDENADAS' if SORT_ANOMALY_SCORE else ''} EN '{OUTPUT_IF_CSV}'")
else:
    print(f"[ WARNING ] Archivo no encontrado: {INPUT_CSV}. Se omite IF global.")
    df = None

# -----------------------------
# ISOLATION FOREST SOBRE DATASET NO CONTAMINADO
# -----------------------------
if os.path.exists(INPUT_UNCONTAMINATED_CSV):
    df_uncont = pd.read_csv(INPUT_UNCONTAMINATED_CSV)
    if SHOW_INFO:
        print(f"[ INFO ] DATASET NO CONTAMINADO CARGADO: {df_uncont.shape[0]} FILAS, {df_uncont.shape[1]} COLUMNAS")

    num_cols = df_uncont.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler_uncont = StandardScaler()
    df_uncont_scaled = scaler_uncont.fit_transform(df_uncont[num_cols])

    clf_uncont = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=VERBOSE
    )
    clf_uncont.fit(df_uncont_scaled)

    anomaly_score_uncont = clf_uncont.decision_function(df_uncont_scaled) * -1
    pred_uncont = clf_uncont.predict(df_uncont_scaled)
    df_uncont['anomaly'] = np.where(pred_uncont == 1, 0, 1)
    df_uncont['anomaly_score'] = anomaly_score_uncont

    if SHOW_INFO:
        print(f"[ INFO ] REGISTROS TOTALES (UNCONTAMINATED): {df_uncont.shape[0]}")
        print(f"[ INFO ] ANOMALÍAS DETECTADAS (UNCONTAMINATED): {df_uncont['anomaly'].sum()}")
        print(f"[ INFO ] PORCENTAJE ANOMALÍAS: {df_uncont['anomaly'].sum()/df_uncont.shape[0]*100:.2f}%")

    df_uncont.to_csv(OUTPUT_UNCONTAMINATED_IF_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] RESULTADOS IF SIN CONTAMINACIÓN EN '{OUTPUT_UNCONTAMINATED_IF_CSV}'")
else:
    df_uncont = None
    print(f"[ WARNING ] Archivo no encontrado: {INPUT_UNCONTAMINATED_CSV}. Se omite IF no contaminado.")

# -----------------------------
# AÑADIR COLUMNAS 'genuine_anomaly' Y 'genuine_anomaly_score'
# -----------------------------
if df is not None and os.path.exists(OUTPUT_UNCONTAMINATED_IF_CSV):
    df_uncont_ref = pd.read_csv(OUTPUT_UNCONTAMINATED_IF_CSV)
    if 'anomaly' in df_uncont_ref.columns and 'anomaly_score' in df_uncont_ref.columns and len(df_uncont_ref) == len(df):
        df['genuine_anomaly'] = df_uncont_ref['anomaly'].values
        df['genuine_anomaly_score'] = df_uncont_ref['anomaly_score'].values
        df.to_csv(OUTPUT_CSV, index=False)
        if SHOW_INFO:
            print(f"[ INFO ] Columnas 'genuine_anomaly' y 'genuine_anomaly_score' añadidas a '{OUTPUT_CSV}'")
    else:
        print("[ WARNING ] No se pudo añadir columnas genuinas: longitud o columnas no coinciden")
else:
    print("[ INFO ] No se añadieron columnas genuinas porque no existe el archivo sin contaminación o se omitió IF global.")

# -----------------------------
# ISOLATION FOREST POR CLUSTER Y ARCHIVOS CONTAMINADOS
# -----------------------------
# Combina archivos por cluster y archivos contaminated_*
files = glob.glob(os.path.join(EXECUTION_FOLDER, CLUSTER_PATTERN)) + \
        glob.glob(os.path.join(EXECUTION_FOLDER, 'contaminated_*.csv'))
# FILTRAR ARCHIVOS QUE TERMINAN EN '_if.csv'
files = [f for f in files if not f.endswith('_if.csv')]

if SHOW_INFO:
    print(f"[ INFO ] ARCHIVOS ENCONTRADOS PARA IF POR CLUSTER/CONTAMINADOS: {len(files)}")

for file_path in files:
    df = pd.read_csv(file_path)
    if SHOW_INFO:
        print(f"[ INFO ] PROCESANDO {file_path}: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

    if 'is_anomaly' in df.columns:
        is_anomaly_column = df['is_anomaly']
        df_input = df.drop(columns=[col for col in ['is_anomaly', 'cluster'] if col in df.columns])
    else:
        is_anomaly_column = pd.Series([0] * len(df), name='is_anomaly')
        df_input = df.copy()

    num_cols = df_input.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        if SHOW_INFO:
            print("[ SKIP ] NO HAY COLUMNAS NUMÉRICAS PARA APLICAR IF")
        continue

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_input[num_cols])

    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=VERBOSE
    )
    clf.fit(df_scaled)

    anomaly_score = clf.decision_function(df_scaled) * -1
    pred = clf.predict(df_scaled)
    df['anomaly'] = np.where(pred == 1, 0, 1)
    df['anomaly_score'] = anomaly_score
    df['is_anomaly'] = is_anomaly_column

    num_anomalies = df['anomaly'].sum()
    num_normals = df.shape[0] - num_anomalies
    if SHOW_INFO:
        print(f"[ INFO ] REGISTROS TOTALES: {df.shape[0]}")
        print(f"[ INFO ] ANOMALÍAS DETECTADAS: {num_anomalies}")
        print(f"[ INFO ] REGISTROS NORMALES: {num_normals}")
        print(f"[ INFO ] PORCENTAJE ANOMALÍAS: {num_anomalies/df.shape[0]*100:.2f}%")

    df.to_csv(file_path, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] RESULTADOS IF COMPLETOS EN {file_path}")

    if SAVE_ANOMALY_CSV:
        df_anomalies = df[df['anomaly'] == 1].copy()
        if NORMALIZE_SCORE:
            min_score = df_anomalies['anomaly_score'].min()
            max_score = df_anomalies['anomaly_score'].max()
            if max_score > min_score:
                df_anomalies['anomaly_score'] = (df_anomalies['anomaly_score'] - min_score) / (max_score - min_score)
        if SORT_ANOMALY_SCORE:
            df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)
        if not INCLUDE_SCORE:
            df_anomalies.drop(columns=['anomaly_score'], inplace=True)
        output_anomaly_csv = file_path.replace('.csv', '_if.csv')
        df_anomalies.to_csv(output_anomaly_csv, index=False)
        if SHOW_INFO:
            print(f"[ GUARDADO ] CSV DE ANOMALÍAS ORDENADAS EN {output_anomaly_csv}")

# -----------------------------
# ISOLATION FOREST PARA 03_GLOBAL_PREDICTIVE_*
# -----------------------------
predictive_files = glob.glob(os.path.join(EXECUTION_FOLDER, '03_global_predictive_*.csv'))
if SHOW_INFO:
    print(f"[ INFO ] ARCHIVOS ENCONTRADOS PARA IF PREDICTIVE: {len(predictive_files)}")

for file_path in predictive_files:
    df = pd.read_csv(file_path)
    if SHOW_INFO:
        print(f"[ INFO ] PROCESANDO {file_path}: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

    if 'is_anomaly' in df.columns:
        is_anomaly_column = df['is_anomaly']
        df_input = df.drop(columns=[col for col in ['is_anomaly', 'cluster'] if col in df.columns])
    else:
        is_anomaly_column = pd.Series([0] * len(df), name='is_anomaly')
        df_input = df.copy()

    num_cols = df_input.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        if SHOW_INFO:
            print("[ SKIP ] NO HAY COLUMNAS NUMÉRICAS PARA APLICAR IF")
        continue

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_input[num_cols])

    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=VERBOSE
    )
    clf.fit(df_scaled)

    anomaly_score = clf.decision_function(df_scaled) * -1
    pred = clf.predict(df_scaled)
    df['anomaly'] = np.where(pred == 1, 0, 1)
    df['anomaly_score'] = anomaly_score
    df['is_anomaly'] = is_anomaly_column

    num_anomalies = df['anomaly'].sum()
    num_normals = df.shape[0] - num_anomalies
    if SHOW_INFO:
        print(f"[ INFO ] REGISTROS TOTALES: {df.shape[0]}")
        print(f"[ INFO ] ANOMALÍAS DETECTADAS: {num_anomalies}")
        print(f"[ INFO ] REGISTROS NORMALES: {num_normals}")
        print(f"[ INFO ] PORCENTAJE ANOMALÍAS: {num_anomalies/df.shape[0]*100:.2f}%")

    # Sobrescribir el archivo original con resultados IF
    df.to_csv(file_path, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] RESULTADOS IF COMPLETOS EN {file_path}")



# -----------------------------
# AÑADIR COLUMNA 'doble_anomaly' A LOS ARCHIVOS CLUSTER_*
# -----------------------------
global_csv_path = os.path.join(EXECUTION_FOLDER, '04_global.csv')

if os.path.exists(global_csv_path):
    df_global = pd.read_csv(global_csv_path, skip_blank_lines=True)
    if 'anomaly' in df_global.columns:
        cluster_files = [f for f in glob.glob(os.path.join(EXECUTION_FOLDER, 'cluster_*.csv')) if not f.endswith('_if.csv')]
        if SHOW_INFO:
            print(f"[ INFO ] AÑADIENDO 'doble_anomaly' A {len(cluster_files)} ARCHIVOS DE CLUSTER...")

        for cluster_file in cluster_files:
            try:
                df_cluster = pd.read_csv(cluster_file, skip_blank_lines=True)

                # Alinear por índice (posición), ignorando posibles diferencias menores
                min_len = min(len(df_cluster), len(df_global))
                df_cluster = df_cluster.iloc[:min_len].copy()
                df_global_part = df_global.iloc[:min_len]

                df_cluster['doble_anomaly'] = np.where(
                    (df_cluster['anomaly'] == 1) & (df_global_part['anomaly'] == 1), 1, 0
                )

                df_cluster.to_csv(cluster_file, index=False)
                if SHOW_INFO:
                    print(f"[ OK ] Añadida columna 'doble_anomaly' en {os.path.basename(cluster_file)} (filas={len(df_cluster)})")

            except Exception as e:
                print(f"[ ERROR ] No se pudo procesar {cluster_file}: {e}")
    else:
        print("[ WARNING ] El archivo global no contiene columna 'anomaly'. No se añade 'doble_anomaly'.")
else:
    print(f"[ WARNING ] No se encontró el archivo global en '{global_csv_path}'. No se añade 'doble_anomaly'.")
