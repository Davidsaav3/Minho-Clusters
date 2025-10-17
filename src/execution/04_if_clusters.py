import pandas as pd
from sklearn.ensemble import IsolationForest  # DETECCIÓN DE ANOMALÍAS
import glob
import os

# CREAR CARPETA DE RESULTADOS
os.makedirs('../../results', exist_ok=True)
print("[ INFO ] Carpeta '../../results' creada si no existía")

# ARCHIVOS DE CLUSTERS
files = glob.glob('../../results/execution/cluster_*.csv')

# PROCESAR CADA ARCHIVO
for file_path in files:
    df = pd.read_csv(file_path)
    print(f"[ INFO ] Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")

    # COLUMNAS NUMÉRICAS PARA IF
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(num_cols) == 0:
        print("[ SKIP ] No hay columnas numéricas para aplicar IF")
        continue
    print(f"[ INFO ] Columnas numéricas: {len(num_cols)}")

    # CONFIGURAR ISOLATION FOREST
    # HIPERPARÁMETROS:
    # n_estimators  -> NÚMERO DE ÁRBOLES EN EL BOSQUE (más árboles = más estabilidad, más tiempo)
    # max_samples   -> MUESTRAS A USAR POR ÁRBOL (int o float). 'auto' = todas o sqrt(n_samples) si muy grande
    # contamination -> PROPORCIÓN ESTIMADA DE ANOMALÍAS (0-0.5), afecta al threshold de decisión
    # max_features  -> NÚMERO DE CARACTERÍSTICAS POR NODO (1.0 = todas)
    # bootstrap     -> SI True, usar muestreo con reemplazo
    # n_jobs        -> NÚMERO DE CORES PARA ENTRENAR (-1 = todos)
    # random_state  -> SEMILLA PARA REPRODUCIBILIDAD
    # verbose       -> NIVEL DE SALIDA EN CONSOLA (0 = sin salida)

    # CONFIGURAR ISOLATION FOREST
    clf = IsolationForest(
        n_estimators=100,       # 100 ÁRBOLES
        max_samples='auto',     # TODAS LAS MUESTRAS
        contamination=0.01,     # 1% ANOMALÍAS
        max_features=1.0,       # TODAS LAS COLUMNAS
        bootstrap=False,        # SIN REEMPLAZO
        n_jobs=-1,              # TODOS LOS NÚCLEOS
        random_state=42,        # REPRODUCIBLE
        verbose=0               # SIN SALIDA EXTRA
    )

    # APLICAR IF
    df['anomaly'] = clf.fit_predict(df[num_cols])
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1=ANOMALÍA, 0=NORMAL

    # CONTAR ANOMALÍAS Y SECUNCIAS
    total_anomalies = df['anomaly'].sum()
    total_normals = df.shape[0] - total_anomalies
    print(f"[ INFO ] Anomalías detectadas: {total_anomalies}")
    print(f"[ INFO ] Registros normales: {total_normals}")
    print(f"[ INFO ] Porcentaje de anomalías: {total_anomalies/df.shape[0]*100:.2f}%")

    # GUARDAR RESULTADO
    df.to_csv(file_path, index=False)
    print(f"[ GUARDADO ] Resultados IF en {file_path}")
