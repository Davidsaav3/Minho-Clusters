import pandas as pd
from sklearn.ensemble import IsolationForest
import os

# PARÁMETROS
RESULTS_FOLDER = '../../results'                      # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DONDE SE GUARDAN RESULTADOS DE EJECUCIÓN
INPUT_CSV = '../../results/execution/00_contaminated.csv'     # CSV DE ENTRADA CON DATOS POSIBLEMENTE ANÓMALOS
OUTPUT_CSV = os.path.join(EXECUTION_FOLDER, 'if_global.csv')  # CSV CON TODOS LOS REGISTROS Y PREDICCIONES
OUTPUT_IF_CSV = os.path.join(EXECUTION_FOLDER, '01_if.csv')   # CSV SOLO CON FILAS DETECTADAS COMO ANOMALÍAS

N_ESTIMATORS = 100          # NÚMERO DE ÁRBOLES EN EL BOSQUE
MAX_SAMPLES = 'auto'        # NÚMERO DE MUESTRAS POR ÁRBOL, 'auto' USA TODAS
CONTAMINATION = 0.01        # FRACCIÓN ESTIMADA DE ANOMALÍAS PARA AJUSTAR EL MODELO
MAX_FEATURES = 1.0          # PROPORCIÓN DE CARACTERÍSTICAS A USAR POR ÁRBOL
BOOTSTRAP = False           # NO USAR MUESTREO CON REPETICIÓN
N_JOBS = -1                 # USAR TODOS LOS HILOS DISPONIBLES
RANDOM_STATE = 42           # SEMILLA PARA REPRODUCIBILIDAD
VERBOSE = 0                 # NIVEL DE VERBOSIDAD
SHOW_INFO = True             # MOSTRAR INFORMACIÓN DE PROGRESO

# CREAR CARPETAS SI NO EXISTEN
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(EXECUTION_FOLDER, exist_ok=True)
if SHOW_INFO:
    print(f"[ INFO ] CARPETAS CREADAS SI NO EXISTÍAN")

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# SEPARAR COLUMNA 'is_anomaly' PARA NO USARLA EN EL MODELO
if 'is_anomaly' in df.columns:
    df_input = df.drop(columns=['is_anomaly'])  # EL MODELO NO DEBE USAR LA ETIQUETA REAL
    is_anomaly_column = df['is_anomaly']       # GUARDAR COLUMNA ORIGINAL PARA POSTERIOR COMPARACIÓN
else:
    df_input = df.copy()                        # SI NO EXISTE, USAR TODO EL DATASET
    is_anomaly_column = pd.Series([0]*len(df_input), name='is_anomaly')  # COLUMNA DE CEROS

# CONFIGURAR EL MODELO ISOLATION FOREST
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

# ENTRENAR EL MODELO
clf.fit(df_input)  # AJUSTAR ISOLATION FOREST A LOS DATOS

# CALCULAR SCORE DE ANOMALÍA
anomaly_score = clf.decision_function(df_input) * -1  # MÁS POSITIVO = MÁS ANÓMALO

# PREDECIR ANOMALÍAS
df['anomaly'] = clf.predict(df_input)                 # 1 = NORMAL, -1 = ANOMALÍA
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})     # CONVERTIR A 0=NORMAL, 1=ANOMALÍA

# AÑADIR SCORE Y COLUMNA ORIGINAL
df['anomaly_score'] = anomaly_score                   # SCORE DE ANOMALÍA
df['is_anomaly'] = is_anomaly_column                 # ETIQUETA ORIGINAL

# INFORMACIÓN GENERAL SOBRE ANOMALÍAS
num_anomalies = df['anomaly'].sum()                  # NÚMERO DE ANOMALÍAS DETECTADAS
num_normals = df.shape[0] - num_anomalies           # NÚMERO DE REGISTROS NORMALES
if SHOW_INFO:
    print(f"[ INFO ] REGISTROS TOTALES: {df.shape[0]}")
    print(f"[ INFO ] ANOMALÍAS DETECTADAS: {num_anomalies}")
    print(f"[ INFO ] REGISTROS NORMALES: {num_normals}")
    print(f"[ INFO ] PORCENTAJE ANOMALÍAS: {num_anomalies/df.shape[0]*100:.2f}%")

# GUARDAR CSV COMPLETO CON PREDICCIONES
df.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] CSV COMPLETO CON ANOMALÍAS EN '{OUTPUT_CSV}'")

# GUARDAR CSV SOLO CON ANOMALÍAS ORDENADAS POR SCORE
df_anomalies = df[df['anomaly'] == 1].copy()        # FILTRAR SOLO ANOMALÍAS
df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False)  # ORDENAR POR SCORE
df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] CSV ANOMALÍAS ORDENADAS EN '{OUTPUT_IF_CSV}'")
