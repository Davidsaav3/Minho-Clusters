import pandas as pd  # PARA MANEJO DE DATAFRAMES
import json          # PARA GUARDAR INFORMACIÓN DE CAMBIOS DE COLUMNAS

# PARÁMETROS CONFIGURABLES
INPUT_FILE = '../../results/preparation/02_nulls.csv'         # RUTA DEL DATASET DE ENTRADA
OUTPUT_FILE = '../../results/preparation/03_codification.csv' # RUTA DEL DATASET CODIFICADO
AUX_FILE = '../../results/preparation/03_aux.json'           # JSON PARA GUARDAR CAMBIOS REALIZADOS
FECHA_COLS = [                                               # COLUMNAS DE FECHA/HORA A EXCLUIR DEL ONE-HOT
    'datetime', 'date', 'time',
    'aemet_hora_minima_temperatura',
    'aemet_hora_maxima_temperatura',
    'aemet_hora_maxima_racha',
    'aemet_hora_maxima_humedad_',
    'aemet_hora_minima_humedad'
]
ONE_HOT_DROP_FIRST = True  # TRUE = ELIMINA LA PRIMERA CATEGORÍA PARA EVITAR MULTICOLINEALIDAD
REPLACE_SPACES = '-'       # CARACTER PARA REEMPLAZAR ESPACIOS EN NUEVAS COLUMNAS

# CARGAR DATASET
df = pd.read_csv(INPUT_FILE)  # LEER CSV DE ENTRADA
print(f"[ INFO ] Dataset cargado: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# INICIALIZAR DICCIONARIO PARA GUARDAR CAMBIOS
columns_changes = {
    'fecha_to_timestamp': [],     # COLUMNAS DE FECHA CONVERTIDAS A TIMESTAMP
    'hora_to_seconds': [],        # COLUMNAS DE HORA CONVERTIDAS A SEGUNDOS
    'one_hot': [],                # COLUMNAS ONE-HOT ENCODED
    'num_imputed_median': []      # COLUMNAS NUMÉRICAS IMPUTADAS CON MEDIANA
}

# CONVERTIR COLUMNAS DE FECHA A TIMESTAMP
for col in ['datetime', 'date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')  # A DATETIME
        df[col] = df[col].astype('int64') // 10**9                         # A SEGUNDOS DESDE 1970
        columns_changes['fecha_to_timestamp'].append(col)
print("[ INFO ] Columnas de fecha convertidas a segundos desde 1970")

# CONVERTIR COLUMNAS DE HORA A SEGUNDOS DESDE MEDIANOCHE
hora_cols = [c for c in FECHA_COLS if c not in ['datetime', 'date']]
for col in hora_cols:
    if col in df.columns:
        df[col] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds()  # A SEGUNDOS
        columns_changes['hora_to_seconds'].append(col)
print("[ INFO ] Columnas de hora convertidas a segundos desde medianoche")

# DETECTAR COLUMNAS CATEGÓRICAS PARA ONE-HOT
cat_cols = df.select_dtypes(include=['object']).columns.tolist()  # COLUMNAS TIPO OBJETO
cat_cols = [c for c in cat_cols if c not in FECHA_COLS]           # EXCLUIR FECHAS/HORAS

# APLICAR ONE-HOT ENCODING
for col in cat_cols:
    if col in df.columns:
        # CREAR DUMMIES Y REEMPLAZAR ESPACIOS EN NOMBRES
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=ONE_HOT_DROP_FIRST)
        dummies.columns = [c.replace(" ", REPLACE_SPACES) for c in dummies.columns]
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
columns_changes['one_hot'] = cat_cols
print("[ INFO ] One-Hot Encoding aplicado (espacios reemplazados por '-')")

# CONVERTIR COLUMNAS NUMÉRICAS Y RELLENAR NaN CON MEDIANA
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    if df[col].notna().sum() == 0:
        df.drop(columns=[col], inplace=True)  # ELIMINAR COLUMNAS VACÍAS COMPLETAMENTE
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())  # IMPUTAR NULOS CON MEDIANA
            columns_changes['num_imputed_median'].append(col)
print("[ INFO ] Columnas numéricas convertidas y NaN imputados con mediana")

# GUARDAR DATASET CODIFICADO FINAL
df.to_csv(OUTPUT_FILE, index=False)
print(f"[ GUARDADO ] Dataset codificado en '{OUTPUT_FILE}'")

# GUARDAR JSON CON CAMBIOS REALIZADOS
with open(AUX_FILE, 'w') as f:
    json.dump(columns_changes, f, indent=4)
print(f"[ GUARDADO ] Cambios de columnas guardados en '{AUX_FILE}'")
