import pandas as pd # de mas anomalo a menos las columnas 
from sklearn.ensemble import IsolationForest  # DETECCIÓN ANOMALÍAS
import os

# PARÁMETROS CONFIGURABLES
RESULTS_FOLDER = '../../results'                        # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN
INPUT_CSV = '../../results/preparation/00_contaminated.csv'       # DATASET DE ENTRADA
OUTPUT_CSV = os.path.join(EXECUTION_FOLDER, 'if_global.csv')  # CSV DE SALIDA

N_ESTIMATORS = 100                                     # NÚMERO DE ÁRBOLES
MAX_SAMPLES = 'auto'                                   # MUESTRAS POR ÁRBOL
CONTAMINATION = 0.01                                   # PROPORCIÓN ESTIMADA DE ANOMALÍAS
MAX_FEATURES = 1.0                                     # CARACTERÍSTICAS POR NODO
BOOTSTRAP = False                                      # MUESTREO CON REEMPLAZO
N_JOBS = -1                                            # NÚMERO DE NÚCLEOS PARA ENTRENAMIENTO
RANDOM_STATE = 42                                      # SEMILLA PARA REPRODUCIBILIDAD
VERBOSE = 0                                            # NIVEL DE SALIDA EN CONSOLA
SHOW_INFO = True                                       # MOSTRAR INFO EN PANTALLA

# CREAR CARPETAS DE RESULTADOS
os.makedirs(RESULTS_FOLDER, exist_ok=True)             # CREAR CARPETA PRINCIPAL SI NO EXISTE
os.makedirs(EXECUTION_FOLDER, exist_ok=True)          # CREAR CARPETA DE EJECUCIÓN SI NO EXISTE
if SHOW_INFO:
    print(f"[ INFO ] Carpetas creadas si no existían")

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)                            # CARGAR CSV DE VARIANZA
if SHOW_INFO:
    print(f"[ INFO ] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

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

# APLICAR ISOLATION FOREST
df['anomaly'] = clf.fit_predict(df)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1=ANOMALÍA, 0=NORMAL

# CONTAR ANOMALÍAS Y NORMALES
num_anomalies = df['anomaly'].sum()           # TOTAL ANOMALÍAS DETECTADAS
num_normals = df.shape[0] - num_anomalies           # TOTAL REGISTROS NORMALES
if SHOW_INFO:
    print(f"[ INFO ] Registros totales: {df.shape[0]}")
    print(f"[ INFO ] Anomalías detectadas: {num_anomalies}")
    print(f"[ INFO ] Registros normales: {num_normals}")
    print(f"[ INFO ] Porcentaje de anomalías: {num_anomalies/df.shape[0]*100:.2f}%")

# GUARDAR RESULTADOS
df.to_csv(OUTPUT_CSV, index=False)                   # GUARDAR CSV CON ANOMALÍAS
if SHOW_INFO:
    print(f"[ GUARDADO ] IF en '{OUTPUT_CSV}'")
