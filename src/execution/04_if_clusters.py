import pandas as pd
from sklearn.ensemble import IsolationForest  # DETECCIÓN DE ANOMALÍAS
import glob
import os

# PARÁMETROS CONFIGURABLES
RESULTS_FOLDER = '../../results'                     # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN
CLUSTER_PATTERN = 'cluster_*.csv'                   # PATRÓN PARA ARCHIVOS DE CLUSTERS

N_ESTIMATORS = 100                                  # NÚMERO DE ÁRBOLES EN EL BOSQUE
MAX_SAMPLES = 'auto'                                # MUESTRAS POR ÁRBOL ('auto' = todas)
CONTAMINATION = 0.01                                # PROPORCIÓN ESTIMADA DE ANOMALÍAS
MAX_FEATURES = 1.0                                  # NÚMERO DE CARACTERÍSTICAS POR NODO
BOOTSTRAP = False                                   # USAR MUESTREO CON REEMPLAZO
N_JOBS = -1                                        # NÚMERO DE CORES PARA ENTRENAR
RANDOM_STATE = 42                                   # SEMILLA PARA REPRODUCIBILIDAD
VERBOSE = 0                                         # NIVEL DE SALIDA EN CONSOLA
SHOW_INFO = True                                    # MOSTRAR INFO EN PANTALLA

# CREAR CARPETAS DE RESULTADOS
os.makedirs(RESULTS_FOLDER, exist_ok=True)         # CREAR CARPETA PRINCIPAL SI NO EXISTE
os.makedirs(EXECUTION_FOLDER, exist_ok=True)      # CREAR CARPETA DE EJECUCIÓN SI NO EXISTE
if SHOW_INFO:
    print(f"[ INFO ] Carpeta '{RESULTS_FOLDER}' creada si no existía")

# LISTAR ARCHIVOS DE CLUSTERS
files = glob.glob(os.path.join(EXECUTION_FOLDER, CLUSTER_PATTERN))

# PROCESAR CADA ARCHIVO
for file_path in files:
    df = pd.read_csv(file_path)
    if SHOW_INFO:
        print(f"[ INFO ] Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")

    # COLUMNAS NUMÉRICAS PARA IF
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        if SHOW_INFO:
            print("[ SKIP ] No hay columnas numéricas para aplicar IF")
        continue
    if SHOW_INFO:
        print(f"[ INFO ] Columnas numéricas: {len(num_cols)}")

    # CONFIGURAR ISOLATION FOREST
    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,       # NÚMERO DE ÁRBOLES
        max_samples=MAX_SAMPLES,         # MUESTRAS POR ÁRBOL
        contamination=CONTAMINATION,     # PROPORCIÓN ESTIMADA DE ANOMALÍAS
        max_features=MAX_FEATURES,       # CARACTERÍSTICAS POR NODO
        bootstrap=BOOTSTRAP,             # USAR MUESTREO CON REEMPLAZO
        n_jobs=N_JOBS,                   # NÚCLEOS PARA ENTRENAMIENTO
        random_state=RANDOM_STATE,       # SEMILLA PARA REPRODUCIBILIDAD
        verbose=VERBOSE                  # NIVEL DE SALIDA
    )

    # APLICAR IF
    df['anomaly'] = clf.fit_predict(df[num_cols])
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1=ANOMALÍA, 0=NORMAL

    # CONTAR ANOMALÍAS Y SECUNCIAS
    total_anomalies = df['anomaly'].sum()
    total_normals = df.shape[0] - total_anomalies
    if SHOW_INFO:
        print(f"[ INFO ] Anomalías detectadas: {total_anomalies}")
        print(f"[ INFO ] Registros normales: {total_normals}")
        print(f"[ INFO ] Porcentaje de anomalías: {total_anomalies/df.shape[0]*100:.2f}%")

    # GUARDAR RESULTADO
    df.to_csv(file_path, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] Resultados IF en {file_path}")
