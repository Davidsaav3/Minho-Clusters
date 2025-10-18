import pandas as pd                                      # IMPORTAR PANDAS PARA MANEJO DE DATAFRAMES
from sklearn.ensemble import IsolationForest            # IMPORTAR ISOLATION FOREST PARA DETECCIÓN DE ANOMALÍAS
import glob                                              # IMPORTAR GLOB PARA BUSCAR ARCHIVOS POR PATRÓN
import os                                                # IMPORTAR OS PARA MANEJO DE RUTAS Y CARPETAS

# PARÁMETROS CONFIGURABLES DE ISOLATION FOREST
RESULTS_FOLDER = '../../results'                        # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  # CARPETA DE EJECUCIÓN PARA ARCHIVOS
CLUSTER_PATTERN = 'cluster_*.csv'                      # PATRÓN DE ARCHIVOS DE CLUSTERS A PROCESAR

# HIPERPARÁMETROS DE ISOLATION FOREST
N_ESTIMATORS = 100                                      # NÚMERO DE ÁRBOLES EN EL BOSQUE (DEFAULT=100)
MAX_SAMPLES = 'auto'                                    # MUESTRAS POR ÁRBOL ('auto' = TODAS O 256) (DEFAULT='auto')
CONTAMINATION = 0.01                                    # FRACCIÓN ESTIMADA DE ANOMALÍAS (DEFAULT='auto')
MAX_FEATURES = 1.0                                      # PROPORCIÓN DE COLUMNAS USADAS POR NODO (DEFAULT=1.0)
BOOTSTRAP = False                                       # USAR MUESTREO CON REEMPLAZO (DEFAULT=False)
N_JOBS = -1                                             # CORES PARA ENTRENAR (-1 = TODOS LOS CORES) (DEFAULT=None)
RANDOM_STATE = 42                                       # SEMILLA PARA REPRODUCIBILIDAD (DEFAULT=None)
VERBOSE = 0                                             # NIVEL DE SALIDA DEL MODELO (DEFAULT=0)
SHOW_INFO = True                                        # MOSTRAR INFORMACIÓN EN CONSOLA

# CREAR CARPETAS SI NO EXISTEN
os.makedirs(RESULTS_FOLDER, exist_ok=True)             # CREAR CARPETA PRINCIPAL
os.makedirs(EXECUTION_FOLDER, exist_ok=True)           # CREAR CARPETA DE EJECUCIÓN
if SHOW_INFO:
    print(f"[ INFO ] CARPETA '{RESULTS_FOLDER}' CREADA SI NO EXISTÍA")  # INFORMAR AL USUARIO

# LISTAR ARCHIVOS QUE CUMPLEN EL PATRÓN
files = glob.glob(os.path.join(EXECUTION_FOLDER, CLUSTER_PATTERN))  # OBTENER TODOS LOS CSV DE CLUSTERS
if SHOW_INFO:
    print(f"[ INFO ] ARCHIVOS ENCONTRADOS PARA IF: {len(files)}")  # MOSTRAR NÚMERO DE ARCHIVOS

# PROCESAR CADA ARCHIVO
for file_path in files:
    df = pd.read_csv(file_path)                          # CARGAR CSV EN DATAFRAME
    if SHOW_INFO:
        print(f"[ INFO ] PROCESANDO {file_path}: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")  # INFORMAR DIMENSIONES

    # SELECCIONAR COLUMNAS NUMÉRICAS PARA IF
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()  # FILTRAR COLUMNAS NUMÉRICAS
    if len(num_cols) == 0:
        if SHOW_INFO:
            print("[ SKIP ] NO HAY COLUMNAS NUMÉRICAS PARA APLICAR IF")  # OMITIR ARCHIVO SI NO HAY COLUMNAS NUMÉRICAS
        continue
    if SHOW_INFO:
        print(f"[ INFO ] NÚMERO DE COLUMNAS NUMÉRICAS: {len(num_cols)}")  # INFORMAR CANTIDAD DE COLUMNAS NUMÉRICAS

    # CONFIGURAR ISOLATION FOREST
    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,                     # ESTABLECER NÚMERO DE ÁRBOLES
        max_samples=MAX_SAMPLES,                       # ESTABLECER NÚMERO DE MUESTRAS POR ÁRBOL
        contamination=CONTAMINATION,                   # FRACCIÓN DE ANOMALÍAS
        max_features=MAX_FEATURES,                     # PROPORCIÓN DE CARACTERÍSTICAS POR NODO
        bootstrap=BOOTSTRAP,                           # USAR MUESTREO CON REEMPLAZO
        n_jobs=N_JOBS,                                 # NÚMERO DE CORES
        random_state=RANDOM_STATE,                     # SEMILLA PARA REPRODUCIBILIDAD
        verbose=VERBOSE                                # NIVEL DE INFORMACIÓN
    )

    # ENTRENAR ISOLATION FOREST Y PREDECIR ANOMALÍAS
    clf.fit(df[num_cols])                               # AJUSTAR MODELO A LOS DATOS
    df['anomaly'] = clf.predict(df[num_cols])          # PREDECIR ANOMALÍAS (1 NORMAL, -1 ANOMALÍA)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})   # CONVERTIR A 0=NORMAL, 1=ANOMALÍA

    # CALCULAR SCORE DE ANOMALÍA
    df['anomaly_score'] = clf.decision_function(df[num_cols]) * -1  # MÁS ALTO = MÁS ANÓMALO

    # INFORMACIÓN GENERAL
    total_anomalies = df['anomaly'].sum()               # CONTAR ANOMALÍAS DETECTADAS
    total_normals = df.shape[0] - total_anomalies      # CALCULAR REGISTROS NORMALES
    if SHOW_INFO:
        print(f"[ INFO ] ANOMALÍAS DETECTADAS: {total_anomalies}")       # MOSTRAR NÚMERO DE ANOMALÍAS
        print(f"[ INFO ] REGISTROS NORMALES: {total_normals}")           # MOSTRAR NÚMERO DE NORMALES
        print(f"[ INFO ] PORCENTAJE ANOMALÍAS: {total_anomalies/df.shape[0]*100:.2f}%")  # MOSTRAR PORCENTAJE

    # GUARDAR RESULTADO COMPLETO
    df.to_csv(file_path, index=False)                    # SOBRESCRIBIR CSV ORIGINAL CON PREDICCIONES
    if SHOW_INFO:
        print(f"[ GUARDADO ] RESULTADOS IF COMPLETOS EN {file_path}")   # CONFIRMAR GUARDADO

    # CREAR CSV CON SOLO ANOMALÍAS ORDENADAS POR SCORE
    df_anomalies = df[df['anomaly'] == 1].copy()        # FILTRAR SOLO ANOMALÍAS
    df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False)  # ORDENAR POR SCORE
    output_anomaly_csv = file_path.replace('.csv', '_if.csv')  # NOMBRE DEL CSV DE ANOMALÍAS

    # GUARDAR CSV DE ANOMALÍAS
    df_anomalies.to_csv(output_anomaly_csv, index=False)  # GUARDAR CSV DE ANOMALÍAS
    if SHOW_INFO:
        print(f"[ GUARDADO ] CSV DE ANOMALÍAS ORDENADAS EN {output_anomaly_csv}")  # CONFIRMAR GUARDADO
