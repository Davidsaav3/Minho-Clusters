# IMPORTS
import pandas as pd  # MANEJO DE DATAFRAMES
import json  # GUARDAR INFORMACIÓN DE COLUMNAS

# CARGAR DATASET TRATADO
df = pd.read_csv('../../results/preparation/02_nulls.csv')
print(f"[ INFO ] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# DICT PARA GUARDAR COLUMNAS CAMBIADAS
columns_changes = {
    'fecha_to_timestamp': [],
    'hora_to_seconds': [],
    'one_hot': [],
    'num_imputed_median': []
}

# COLUMNAS DE FECHA/HORA QUE NO SE CODIFICAN
fecha_cols = [
    'datetime', 'date', 'time',
    'aemet_hora_minima_temperatura',
    'aemet_hora_maxima_temperatura',
    'aemet_hora_maxima_racha',
    'aemet_hora_maxima_humedad_',
    'aemet_hora_minima_humedad'
]

# CONVERTIR FECHASTIMESTAMP (SEGUNDOS DESDE 1970)
for col in ['datetime', 'date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
        df[col] = df[col].astype('int64') // 10**9
        columns_changes['fecha_to_timestamp'].append(col)
print(f"[ INFO ] Columnas de fecha convertidassegundos desde 1970")

# CONVERTIR HORASSEGUNDOS DESDE MEDIANOCHE
hora_cols = [
    'time', 'aemet_hora_minima_temperatura', 'aemet_hora_maxima_temperatura',
    'aemet_hora_maxima_racha', 'aemet_hora_maxima_humedad_', 'aemet_hora_minima_humedad'
]
for col in hora_cols:
    if col in df.columns:
        df[col] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds()
        columns_changes['hora_to_seconds'].append(col)
print(f"[ INFO ] Columnas de hora convertidassegundos desde medianoche")

# DETECTAR COLUMNAS CATEGÓRICAS EXCLUYENDO FECHAS/HORAS
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in fecha_cols]

# ONE-HOT ENCODING DE COLUMNAS CATEGÓRICAS
for col in cat_cols:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        # REEMPLAZAR ESPACIOS POR GUIONES EN LOS NOMBRES DE COLUMNAS NUEVAS
        dummies.columns = [c.replace(" ", "-") for c in dummies.columns]
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
columns_changes['one_hot'] = cat_cols
print(f"[ INFO ] One-Hot Encoding aplicado (columnas con '-' en lugar de espacios)")

# CONVERTIR COLUMNAS NUMÉRICAS A FLOAT Y RELLENAR NAN CON MEDIANA
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    if df[col].notna().sum() == 0:
        df.drop(columns=[col], inplace=True)  # ELIMINAR COLUMNAS VACÍAS
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            columns_changes['num_imputed_median'].append(col)
print(f"[ INFO ] Columnas numéricas convertidas y valores NaN imputados con mediana")

# GUARDAR DATASET CODIFICADO
df.to_csv('../../results/preparation/03_codification.csv', index=False)
print("[ GUARDADO ] Dataset codificado en '../../results/preparation/03_codification.csv'")

# GUARDAR CAMBIOS DE COLUMNAS EN JSON
with open('../../results/preparation/03_aux.json', 'w') as f:
    json.dump(columns_changes, f, indent=4)
print("[ GUARDADO ] Cambios de columnas guardados en '../../results/preparation/03_aux.json'")
