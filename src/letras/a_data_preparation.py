# ARCHIVO: prepare_dataset_for_iforest.py
# PREPARA DATASET PARA ISOLATION FOREST SIGUIENDO PASOS ESPECÍFICOS

import pandas as pd
import numpy as np
import os
import json
import traceback
from sklearn.preprocessing import StandardScaler

# CARGAR HIPERPARÁMETROS
try:
    with open('hiperparameters.json', 'r') as config_file:
        config = json.load(config_file)['a_data_preparation']
except (FileNotFoundError, KeyError) as e:
    print(f"[ERROR]: NO SE PUDO CARGAR hiperparameters.json: {e}".upper())
    exit(1)

# EXTRAER HIPERPARÁMETROS
input_dataset_path = config['input_dataset_path']
csv_encoding = config['csv_encoding']
results_directory = config['results_directory']
log_file_path = config['log_file_path']
null_counts_output_path = config['null_counts_output_path']
clean_dataset_output_path = config['clean_dataset_output_path']
temporal_columns = config.get('temporal_columns', [])  # Lista opcional de columnas temporales
datetime_column = config['datetime_column']
low_memory = config['low_memory']

# CREAR DIRECTORIO DE RESULTADOS
os.makedirs(results_directory, exist_ok=True)

# INICIAR LOG
with open(log_file_path, 'a') as log_file:
    log_file.write("\n[ prepare_dataset_for_iforest ]\n")

# FUNCIÓN PARA REGISTRAR MENSAJES
def log_message(message):
    message_upper = message.upper()
    print(message_upper)
    with open(log_file_path, 'a') as log_file:
        log_file.write(message_upper + "\n")

# PASO 1: CARGAR DATASET
try:
    dataset = pd.read_csv(input_dataset_path, encoding=csv_encoding, low_memory=low_memory)
    log_message("[CARGADO]: DATASET CARGADO CON ÉXITO.")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO CARGAR CON {csv_encoding}: {e}. INTENTANDO ISO-8859-1...")
    try:
        dataset = pd.read_csv(input_dataset_path, encoding='ISO-8859-1', low_memory=low_memory)
        log_message("[CARGADO]: DATASET CARGADO CON ISO-8859-1.")
    except Exception as e2:
        log_message(f"[ERROR FATAL]: NO SE PUDO CARGAR EL DATASET: {e2}")
        traceback.print_exc()
        raise

log_message(f"-> TAMAÑO INICIAL DEL DATASET: {dataset.shape}")

# PASO 2: TRATAR COLUMNAS TEMPORALES (DATETIME Y OTRAS)
temporal_columns = temporal_columns + [datetime_column] if datetime_column in dataset.columns else temporal_columns
temporal_columns = list(set(temporal_columns))  # Eliminar duplicados

for temp_col in temporal_columns:
    if temp_col in dataset.columns:
        try:
            log_message(f"-> MUESTRA DE VALORES EN {temp_col}: {dataset[temp_col].head(10).tolist()}")
            unique_values = dataset[temp_col].dropna().unique()[:10]
            log_message(f"-> VALORES ÚNICOS EN {temp_col} (MUESTRA): {list(unique_values)}")
            empty_count = dataset[temp_col].isna().sum()
            invalid_count = dataset[temp_col].str.strip().eq('').sum() if dataset[temp_col].dtype == "object" else 0
            log_message(f"-> VALORES VACÍOS O NULOS EN {temp_col} ANTES DE CONVERSIÓN: {empty_count + invalid_count}")

            # Reemplazar valores vacíos o no válidos por NaN
            dataset[temp_col] = dataset[temp_col].replace(['', 'N/A', 'sin datos', 'NaN', 'nan', 'VARIAS', 'FIN DE SEMANA'], np.nan)
            
            # Intentar conversión con múltiples formatos
            if 'HORA' in temp_col.upper() or 'TIME' in temp_col.upper():
                # Columnas de hora (sin fecha)
                dataset[temp_col] = pd.to_datetime(
                    dataset[temp_col], 
                    format='%H:%M:%S',  # 00:15:00
                    errors='coerce'
                ).fillna(
                    pd.to_datetime(
                        dataset[temp_col], 
                        format='%H:%M',  # 00:15
                        errors='coerce'
                    )
                )
                # Extraer características de hora
                dataset[f'{temp_col}_hour'] = dataset[temp_col].dt.hour
                dataset[f'{temp_col}_minute'] = dataset[temp_col].dt.minute
            else:
                # Columnas de fecha o fecha-hora
                dataset[temp_col] = pd.to_datetime(
                    dataset[temp_col], 
                    format='%d/%m/%Y %H:%M:%S',  # 01/01/2021 00:15:00
                    errors='coerce'
                ).fillna(
                    pd.to_datetime(
                        dataset[temp_col], 
                        format='%d/%m/%Y %H:%M',  # 01/01/2021 00:15
                        errors='coerce'
                    )
                ).fillna(
                    pd.to_datetime(
                        dataset[temp_col], 
                        format='%d/%m/%Y',  # 01/01/2021
                        errors='coerce'
                    )
                ).fillna(
                    pd.to_datetime(
                        dataset[temp_col], 
                        format='%Y-%m-%d',  # 2021-01-01
                        errors='coerce'
                    )
                )
                # Extraer características de fecha
                dataset[f'{temp_col}_year'] = dataset[temp_col].dt.year
                dataset[f'{temp_col}_month'] = dataset[temp_col].dt.month
                dataset[f'{temp_col}_day'] = dataset[temp_col].dt.day
                dataset[f'{temp_col}_hour'] = dataset[temp_col].dt.hour
                dataset[f'{temp_col}_minute'] = dataset[temp_col].dt.minute
                dataset[f'{temp_col}_weekday'] = dataset[temp_col].dt.weekday
            
            log_message(f"-> CONVERSIÓN DE {temp_col} COMPLETADA.")
            
            # Verificar valores NaT
            nat_count = dataset[temp_col].isna().sum()
            if nat_count > 0:
                log_message(f"[ADVERTENCIA]: {nat_count} VALORES NAT DETECTADOS EN {temp_col}.")
                invalid_values = dataset[temp_col][dataset[temp_col].isna()].head(10).tolist()
                log_message(f"-> MUESTRA DE VALORES NO CONVERTIDOS EN {temp_col}: {invalid_values}")
            
            # Imputar valores NaN en columnas derivadas
            derived_cols = [col for col in dataset.columns if col.startswith(f'{temp_col}_')]
            for col in derived_cols:
                dataset[col] = dataset[col].fillna(dataset[col].median())
            # Eliminar la columna original
            dataset = dataset.drop(columns=[temp_col])
            log_message(f"-> CARACTERÍSTICAS NUMÉRICAS EXTRAÍDAS DE {temp_col} Y NULOS IMPUTADOS.")
        except Exception as e:
            log_message(f"[ERROR]: FALLÓ LA CONVERSIÓN O EXTRACCIÓN DE {temp_col}: {e}")
            raise
    else:
        log_message(f"[ADVERTENCIA]: COLUMNA '{temp_col}' NO ENCONTRADA.")

# PASO 3: IDENTIFICAR COLUMNAS NUMÉRICAS Y NO NUMÉRICAS
numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
log_message(f"-> COLUMNAS NUMÉRICAS: {len(numeric_cols)}")
non_numeric_cols = dataset.select_dtypes(exclude=[np.number]).columns.tolist()
log_message(f"-> COLUMNAS NO NUMÉRICAS: {len(non_numeric_cols)}")

# PASO 4: GUARDAR COLUMNAS CON NULOS
try:
    null_counts = dataset.isnull().sum()
    nulls_df = null_counts[null_counts > 0].reset_index()
    nulls_df.columns = ['column_name', 'null_count']
    nulls_df['null_rows'] = nulls_df['column_name'].apply(
        lambda col: str(dataset[col][dataset[col].isna()].index.tolist())
    )
    for _, row in nulls_df.iterrows():
        null_rows = eval(row['null_rows'])
        truncated_rows = null_rows[:100]
        log_message(f"-> COLUMNA '{row['column_name']}': {row['null_count']} NULOS EN FILAS {truncated_rows}{'...' if len(null_rows) > 100 else ''}")
    nulls_df.to_csv(null_counts_output_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {null_counts_output_path}")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO GUARDAR {null_counts_output_path}: {e}")
    raise

# PASO 5: TRATAR COLUMNAS NO NUMÉRICAS
try:
    high_cardinality_cols = []
    valid_non_numeric_cols = []
    for col in non_numeric_cols:
        unique_count = dataset[col].nunique()
        unique_values_sample = dataset[col].dropna().unique()[:5].tolist()
        log_message(f"-> COLUMNA NO NUMÉRICA '{col}': {unique_count} VALORES ÚNICOS, MUESTRA: {unique_values_sample}")
        if unique_count > 20:
            high_cardinality_cols.append(col)
        else:
            valid_non_numeric_cols.append(col)
    
    if high_cardinality_cols:
        log_message(f"[ADVERTENCIA]: COLUMNAS CON ALTA CARDINALIDAD EXCLUIDAS DE ONE-HOT ENCODING: {high_cardinality_cols}")
    
    # Guardar columnas no numéricas
    non_numeric_df = pd.DataFrame({
        'non_numeric_column': non_numeric_cols,
        'status': ['excluded (high cardinality)' if col in high_cardinality_cols else 'included' for col in non_numeric_cols]
    })
    non_numeric_output_path = os.path.join(results_directory, 'a_no_numeric_columns.csv')
    non_numeric_df.to_csv(non_numeric_output_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {non_numeric_output_path}")

    # Imputar nulos en columnas no numéricas válidas
    if valid_non_numeric_cols:
        for col in valid_non_numeric_cols:
            dataset[col] = dataset[col].fillna('missing')
        log_message("-> VALORES NULOS EN COLUMNAS NO NUMÉRICAS IMPUTADOS CON 'missing'.")
        
        # Aplicar One-Hot Encoding
        dataset = pd.get_dummies(dataset, columns=valid_non_numeric_cols, prefix=valid_non_numeric_cols)
        log_message(f"-> COLUMNAS NO NUMÉRICAS CODIFICADAS CON ONE-HOT ENCODING: {len(dataset.columns) - len(numeric_cols)} NUEVAS COLUMNAS.")
    
    # Eliminar columnas de alta cardinalidad
    if high_cardinality_cols:
        dataset = dataset.drop(columns=high_cardinality_cols)
        log_message(f"-> COLUMNAS DE ALTA CARDINALIDAD ELIMINADAS: {high_cardinality_cols}")
except Exception as e:
    log_message(f"[ERROR]: FALLÓ EL TRATAMIENTO DE COLUMNAS NO NUMÉRICAS: {e}")
    raise

# PASO 6: TRATAR VALORES NULOS EN COLUMNAS NUMÉRICAS
try:
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        dataset[col] = dataset[col].astype(np.float64)  # Convertir a float64
        dataset[col] = dataset[col].fillna(dataset[col].median())
    log_message("-> VALORES NULOS EN COLUMNAS NUMÉRICAS IMPUTADOS CON LA MEDIANA.")
except Exception as e:
    log_message(f"[ERROR]: FALLÓ LA IMPUTACIÓN DE COLUMNAS NUMÉRICAS: {e}")
    raise

# PASO 7: CREAR DATASET LIMPIO
clean_dataset = dataset.copy()
log_message(f"-> TAMAÑO DEL DATASET LIMPIO: {clean_dataset.shape}")

# PASO 8: ESCALAR VARIABLES NUMÉRICAS
try:
    numeric_cols = clean_dataset.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    clean_dataset.loc[:, numeric_cols] = scaler.fit_transform(clean_dataset[numeric_cols])
    log_message("-> COLUMNAS NUMÉRICAS ESCALADAS CON STANDARDSCALER.")
except Exception as e:
    log_message(f"[ERROR]: FALLÓ EL ESCALADO DE COLUMNAS NUMÉRICAS: {e}")
    raise

# PASO 9: VERIFICACIÓN FINAL
try:
    if np.any(np.isnan(clean_dataset[numeric_cols])):
        log_message("[ERROR]: SE DETECTARON VALORES NAN EN EL DATASET LIMPIO.")
        raise ValueError("Valores NaN detectados en el dataset limpio.")
    if not np.isfinite(clean_dataset[numeric_cols]).all().all():
        log_message("[ERROR]: SE DETECTARON VALORES INFINITOS EN EL DATASET LIMPIO.")
        raise ValueError("Valores infinitos detectados en el dataset limpio.")
    log_message("-> VERIFICACIÓN FINAL: NO SE DETECTARON NAN NI VALORES INFINITOS.")
except Exception as e:
    log_message(f"[ERROR]: FALLÓ LA VERIFICACIÓN FINAL: {e}")
    raise

# PASO 10: GUARDAR DATASET LIMPIO
try:
    clean_dataset.to_csv(clean_dataset_output_path, index=False, encoding=csv_encoding)
    log_message(f"[GUARDADO]: {clean_dataset_output_path}")
except Exception as e:
    log_message(f"[ERROR]: NO SE PUDO GUARDAR {clean_dataset_output_path}: {e}")
    raise