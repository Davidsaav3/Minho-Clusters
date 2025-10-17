# src/03_codificacion.py
import pandas as pd

# Cargar dataset procesado en paso anterior
df = pd.read_csv('../results/02_tratamiento_nulos.csv')

# Columnas de fecha/hora que NO se deben codificar
fecha_cols = [
    'datetime', 'date', 'time',
    'aemet_hora_minima_temperatura',
    'aemet_hora_maxima_temperatura',
    'aemet_hora_maxima_racha',
    'aemet_hora_maxima_humedad_',
    'aemet_hora_minima_humedad'
]

# Convertir columnas de fechas a timestamps numéricos (segundos desde 1970)
for col in ['datetime', 'date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
        df[col] = df[col].astype('int64') // 10**9  # segundos desde 1970

# Convertir columnas de horas a segundos desde medianoche
hora_cols = [
    'time', 'aemet_hora_minima_temperatura', 'aemet_hora_maxima_temperatura',
    'aemet_hora_maxima_racha', 'aemet_hora_maxima_humedad_', 'aemet_hora_minima_humedad'
]
for col in hora_cols:
    if col in df.columns:
        df[col] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds()

# Detectar automáticamente columnas categóricas (object) excluyendo fechas/horas
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in fecha_cols]

# One-Hot Encoding solo de columnas categóricas válidas
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Convertir columnas numéricas a float y rellenar NaN con la mediana
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    # Eliminar columnas completamente vacías
    if df[col].notna().sum() == 0:
        df.drop(columns=[col], inplace=True)
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# Guardar dataset codificado listo para escalado
df.to_csv('../results/03_codificacion.csv', index=False)
print("Dataset codificado guardado en '../results/03_codificacion.csv'")
