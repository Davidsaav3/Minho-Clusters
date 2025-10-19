import pandas as pd
from sklearn.ensemble import IsolationForest
import os

# PARÁMETROS 
RESULTS_FOLDER = '../../results'                      
# CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  
# CARPETA DONDE SE GUARDAN RESULTADOS DE EJECUCIÓN
INPUT_CSV = '../../results/execution/00_contaminated.csv'     
# CSV DE ENTRADA CON DATOS POSIBLEMENTE ANÓMALOS
OUTPUT_CSV = os.path.join(EXECUTION_FOLDER, 'if_global.csv')  
# CSV CON TODOS LOS REGISTROS Y PREDICCIONES
OUTPUT_IF_CSV = os.path.join(EXECUTION_FOLDER, '01_if.csv')   
# CSV SOLO CON FILAS DETECTADAS COMO ANOMALÍAS

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

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# SEPARAR COLUMNA 'is_anomaly' PARA NO USARLA EN EL MODELO
# SI EXISTE, SEPARARLA PARA POSTERIOR COMPARACIÓN
if 'is_anomaly' in df.columns:
    df_input = df.drop(columns=['is_anomaly'])
    is_anomaly_column = df['is_anomaly']
else:
    df_input = df.copy()
    is_anomaly_column = pd.Series([0]*len(df_input), name='is_anomaly')

# CONFIGURAR Y ENTRENAR ISOLATION FOREST
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
clf.fit(df_input)
# ENTRENAMIENTO COMPLETADO

# CALCULAR SCORE DE ANOMALÍA
# MÁS POSITIVO = MÁS ANÓMALO
anomaly_score = clf.decision_function(df_input) * -1  

# PREDECIR ANOMALÍAS
df['anomaly'] = clf.predict(df_input)                 
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})     
# 0 = NORMAL, 1 = ANOMALÍA

# AÑADIR SCORE Y COLUMNA ORIGINAL
df['anomaly_score'] = anomaly_score                   
df['is_anomaly'] = is_anomaly_column                 

# INFORMACIÓN GENERAL SOBRE ANOMALÍAS
num_anomalies = df['anomaly'].sum()
num_normals = df.shape[0] - num_anomalies
if SHOW_INFO:
    print(f"[ INFO ] REGISTROS TOTALES: {df.shape[0]}")
    print(f"[ INFO ] ANOMALÍAS DETECTADAS: {num_anomalies}")
    print(f"[ INFO ] REGISTROS NORMALES: {num_normals}")
    print(f"[ INFO ] PORCENTAJE ANOMALÍAS: {num_anomalies/df.shape[0]*100:.2f}%")

# GUARDAR CSV COMPLETO CON PREDICCIONES
df.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] CSV COMPLETO CON ANOMALÍAS EN '{OUTPUT_CSV}'")

# GUARDAR CSV SOLO CON ANOMALÍAS
if SAVE_ANOMALY_CSV:
    df_anomalies = df[df['anomaly'] == 1].copy()
    if SORT_ANOMALY_SCORE:
        df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False)
    if not INCLUDE_SCORE:
        df_anomalies = df_anomalies.drop(columns=['anomaly_score'])
    df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] CSV ANOMALÍAS {'ORDENADAS' if SORT_ANOMALY_SCORE else ''} EN '{OUTPUT_IF_CSV}'")
