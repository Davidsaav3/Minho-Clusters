import pandas as pd
from sklearn.ensemble import IsolationForest
import glob
import os
import numpy as np

# PARÁMETROS 
RESULTS_FOLDER = '../../results'                        
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')
CLUSTER_PATTERN = 'cluster_*.csv'  # PATRÓN DE ARCHIVOS DE CLUSTERS

SAVE_ANOMALY_CSV = True  
# TRUE = GUARDAR CSV SOLO CON ANOMALÍAS DETECTADAS
SORT_ANOMALY_SCORE = True  
# TRUE = ORDENAR CSV DE ANOMALÍAS POR SCORE (MÁS ANÓMALAS ARRIBA)
INCLUDE_SCORE = True  
# TRUE = INCLUIR COLUMNA 'anomaly_score' EN CSV DE ANOMALÍAS

# HIPERPARÁMETROS ISOLATION FOREST
N_ESTIMATORS = 100
# NÚMERO DE ÁRBOLES EN EL BOSQUE
# MÁS ÁRBOLES = MODELO MÁS ESTABLE Y PRECISO, PERO MÁS LENTO
MAX_SAMPLES = 'auto'
# NÚMERO DE MUESTRAS POR ÁRBOL
# 'auto' USA TODAS LAS FILAS
# MENOS MUESTRAS = ENTRENAMIENTO MÁS RÁPIDO, PERO MENOS ROBUSTO
CONTAMINATION = 0.01
# FRACCIÓN ESTIMADA DE ANOMALÍAS
# AJUSTA UMBRAL PARA CLASIFICAR ANOMALÍAS
# RECOMENDADO: 0.01-0.05 SEGÚN EXPECTATIVA DE ANOMALÍAS
MAX_FEATURES = 1.0
# PROPORCIÓN DE CARACTERÍSTICAS A USAR POR ÁRBOL
# MENOS CARACTERÍSTICAS = MÁS VARIABILIDAD ENTRE ÁRBOLES

BOOTSTRAP = False
# TRUE = MUESTREO CON REPETICIÓN POR ÁRBOL
# FALSE = MUESTRA SIN REPETICIÓN, MÁS PRECISO
N_JOBS = -1
# NÚMERO DE HILOS PARA ENTRENAMIENTO
# -1 USA TODOS LOS HILOS DISPONIBLES
RANDOM_STATE = 42
# SEMILLA PARA REPRODUCIBILIDAD DE RESULTADOS
VERBOSE = 0
# NIVEL DE VERBOSIDAD DEL MODELO
# 0 = SILENCIOSO, >0 = INFORMACIÓN DETALLADA DURANTE ENTRENAMIENTO
SHOW_INFO = True
# TRUE = MOSTRAR INFORMACIÓN DEL PROCESO EN CONSOLA

# LISTAR ARCHIVOS
files = glob.glob(os.path.join(EXECUTION_FOLDER, CLUSTER_PATTERN))  # LISTAR CSV DE CLUSTERS
if SHOW_INFO:
    print(f"[ INFO ] ARCHIVOS ENCONTRADOS PARA IF: {len(files)}")

# PROCESAR CADA ARCHIVO
for file_path in files:
    df = pd.read_csv(file_path)  # CARGAR CSV
    if SHOW_INFO:
        print(f"[ INFO ] PROCESANDO {file_path}: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

    # SEPARAR COLUMNA 'is_anomaly' SI EXISTE
    if 'is_anomaly' in df.columns:
        df_input = df.drop(columns=['is_anomaly'])  # ELIMINAR COLUMNA ORIGINAL
        is_anomaly_column = df['is_anomaly']       # GUARDAR COLUMNA PARA POSTERIOR USO
    else:
        df_input = df.copy()
        is_anomaly_column = pd.Series([0]*len(df_input), name='is_anomaly')  # CREAR COLUMNA FALSA

    # SELECCIONAR COLUMNAS NUMÉRICAS PARA IF
    num_cols = df_input.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        if SHOW_INFO:
            print("[ SKIP ] NO HAY COLUMNAS NUMÉRICAS PARA APLICAR IF")
        continue
    if SHOW_INFO:
        print(f"[ INFO ] NÚMERO DE COLUMNAS NUMÉRICAS: {len(num_cols)}")

    # CONFIGURAR ISOLATION FOREST
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

    # ENTRENAR Y PREDECIR ANOMALÍAS
    clf.fit(df_input[num_cols])                          # ENTRENAMIENTO
    df['anomaly'] = np.where(clf.predict(df_input[num_cols]) == 1, 0, 1)  # PREDICCIÓN 0=NORMAL, 1=ANOMALÍA
    df['anomaly_score'] = clf.decision_function(df_input[num_cols]) * -1  # CALCULAR SCORE
    df['is_anomaly'] = is_anomaly_column               # AÑADIR COLUMNA ORIGINAL

    # INFORMACIÓN GENERAL
    total_anomalies = df['anomaly'].sum()             # CONTAR ANOMALÍAS
    total_normals = df.shape[0] - total_anomalies    # CONTAR NORMALES
    if SHOW_INFO:
        print(f"[ INFO ] ANOMALÍAS DETECTADAS: {total_anomalies}")
        print(f"[ INFO ] REGISTROS NORMALES: {total_normals}")
        print(f"[ INFO ] PORCENTAJE ANOMALÍAS: {total_anomalies/df.shape[0]*100:.2f}%")

    # GUARDAR CSV COMPLETO CON PREDICCIONES
    df.to_csv(file_path, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] RESULTADOS IF COMPLETOS EN {file_path}")

    # GUARDAR CSV SOLO CON ANOMALÍAS
    if SAVE_ANOMALY_CSV:
        df_anomalies = df[df['anomaly'] == 1].copy()  # FILTRAR ANOMALÍAS
        if SORT_ANOMALY_SCORE:
            df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False)  # ORDENAR
        if not INCLUDE_SCORE:
            df_anomalies = df_anomalies.drop(columns=['anomaly_score'])  # ELIMINAR SCORE SI NO SE INCLUYE
        output_anomaly_csv = file_path.replace('.csv', '_if.csv')  # NOMBRE CSV ANOMALÍAS
        df_anomalies.to_csv(output_anomaly_csv, index=False)
        if SHOW_INFO:
            print(f"[ GUARDADO ] CSV DE ANOMALÍAS ORDENADAS EN {output_anomaly_csv}")
