import pandas as pd
from sklearn.ensemble import IsolationForest  # DETECCIÓN ANOMALÍAS
import os

# CREAR CARPETA DE RESULTADOS
os.makedirs('../../results', exist_ok=True)
print("[ INFO ] Carpeta '../../results' creada si no existía")

# CARGAR DATASET
df = pd.read_csv('../../results/preparation/05_variance.csv')
print(f"[ INFO ] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

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

# APLICAR ISOLATION FOREST
df['anomaly_global'] = clf.fit_predict(df)
df['anomaly_global'] = df['anomaly_global'].map({1:0, -1:1})  # 1=ANOMALÍA, 0=NORMAL

# CONTAR ANOMALÍAS Y NORMALES
num_anomalies = df['anomaly_global'].sum()
num_normals = df.shape[0] - num_anomalies
print(f"[ INFO ] Registros totales: {df.shape[0]}")
print(f"[ INFO ] Anomalías detectadas: {num_anomalies}")
print(f"[ INFO ] Registros normales: {num_normals}")
print(f"[ INFO ] Porcentaje de anomalías: {num_anomalies/df.shape[0]*100:.2f}%")

# GUARDAR RESULTADOS
df.to_csv('../../results/execution/if_global.csv', index=False)
print("[ GUARDADO ] IF en '../../results/execution/if_global.csv'")
