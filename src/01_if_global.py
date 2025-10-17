# src/01_if_global.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import os

# =========================
# Crear carpeta de resultados si no existe
# =========================
os.makedirs('../results', exist_ok=True)

# =========================
# Cargar dataset codificado
# =========================
df = pd.read_csv('../results/05_seleccion.csv')

# =========================
# Configuración de Isolation Forest
# =========================

# Hiperparámetros principales:
# n_estimators      : número de árboles en el bosque. Más árboles = mayor estabilidad, pero más lento.
# max_samples       : número de muestras a extraer de X para entrenar cada árbol. Puede ser int o float (fracción del dataset).
# contamination     : proporción estimada de anomalías en los datos (0-0.5), afecta al threshold de decisión.
# max_features      : número de características a considerar al dividir cada nodo. Por defecto 1.0 = todas.
# bootstrap         : si True, usar bootstrap sampling (muestreo con reemplazo) al construir árboles.
# n_jobs            : número de CPUs a usar. -1 usa todos.
# behaviour         : versión de la API de sklearn (deprecated en versiones recientes, se ignora).
# random_state      : semilla para reproducibilidad.
# verbose           : nivel de mensajes en la consola.

clf = IsolationForest(
    n_estimators=100,       # 100 árboles
    max_samples='auto',     # usa todas las muestras o sqrt(n_samples) si es muy grande
    contamination=0.01,     # 1% de los datos se considera anomalía
    max_features=1.0,       # usa todas las columnas para cada árbol
    bootstrap=False,        # no usar muestreo con reemplazo
    n_jobs=-1,              # usar todos los núcleos disponibles
    random_state=42,        # para reproducibilidad
    verbose=0               # sin salida adicional
)

# =========================
# Aplicar Isolation Forest
# =========================
df['anomaly_global'] = clf.fit_predict(df)

# IF devuelve -1 para anomalías, 1 para normales
# Convertimos a 1=anomalía, 0=normal para mayor claridad
df['anomaly_global'] = df['anomaly_global'].map({1: 0, -1: 1})

# =========================
# Contar anomalías y normales
# =========================
num_anomalies = df['anomaly_global'].sum()
num_normals = df.shape[0] - num_anomalies
print(f"Registros totales: {df.shape[0]}")
print(f"Anomalías detectadas: {num_anomalies}")
print(f"Registros normales: {num_normals}")
print(f"Porcentaje de anomalías: {num_anomalies/df.shape[0]*100:.2f}%")

# =========================
# Guardar resultados
# =========================
df.to_csv('../results/01_if_global.csv', index=False)
print("IF global aplicado y guardado en '../results/01_if_global.csv'")
