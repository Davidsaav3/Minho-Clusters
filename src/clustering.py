# FILE: clustering.py
# THIS SCRIPT CREATES MANUAL CLUSTERS BASED ON SEASON AND ENVIRONMENTAL FACTORS LIKE TEMPERATURE; ADAPTED SINCE NO EXPLICIT WELL_ID COLUMN.

import pandas as pd
import os
import traceback

# CHECK IF DATASET WITH CONTINUITY EXISTS
continuity_path = '../results/dataset_with_continuity.csv'
if not os.path.exists(continuity_path):
    raise FileNotFoundError(f"{continuity_path} NOT FOUND. RUN CONTINUITY_ANALYSIS FIRST.")

print(f"LOADING DATASET FROM: {continuity_path}")

# LOAD DATASET WITH CONTINUITY FROM RESULTS FOLDER: Use a robust read method
try:
    # Intento robusto de lectura con la codificación más amplia (ISO-8859-1 / latin1)
    with open(continuity_path, 'r', encoding='ISO-8859-1', errors='replace') as f:
        dataset = pd.read_csv(f, low_memory=False)
    print("Dataset loaded using ISO-8859-1 with errors='replace'.")
except Exception as e:
    print(f"Error loading with ISO-8859-1: {e}. Trying utf-8...")
    # Si falla, intentar la lectura UTF-8 original
    try:
        with open(continuity_path, 'r', encoding='utf-8', errors='replace') as f:
            dataset = pd.read_csv(f, low_memory=False)
        print("Dataset loaded using utf-8 with errors='replace'.")
    except Exception as e2:
        print(f"FATAL ERROR: Failed to load input file. {e2}")
        traceback.print_exc()
        raise


# RENAME COLUMN IF IT HAS BOM PREFIX (IN CASE IT WASN'T SAVED PROPERLY)
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': 'datetime'}, inplace=True)
    print("RENAMED 'ï»¿datetime' TO 'datetime'.")

# CONVERT 'DATETIME' COLUMN TO DATETIME AFTER LOADING IF IT EXISTS
if 'datetime' in dataset.columns:
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], errors='coerce')

# SEASON IS ALREADY IN DATASET; BIN TEMPERATURE FOR CLUSTERING USING 'AEMET_TEMPERATURA_MEDIA' AS ENVIRONMENTAL FACTOR
if 'aemet_temperatura_media' in dataset.columns:
    # Asegurarse de que no hay NaNs en la columna antes de cortar
    temp_data = dataset['aemet_temperatura_media'].dropna()
    if not temp_data.empty:
        # Usa el número de bins que Pandas determine automáticamente (o 3 si se quiere forzar)
        temp_bins = pd.cut(dataset['aemet_temperatura_media'], bins=3, labels=['Low', 'Med', 'High'], duplicates='drop')
    else:
        temp_bins = pd.Series(['Med'] * len(dataset), index=dataset.index)
else:
    # FALLBACK TO ANOTHER TEMP COLUMN IF NEEDED
    temp_col = next((col for col in dataset.columns if 'temperatura' in col.lower()), None)
    if temp_col and not dataset[temp_col].isnull().all():
        temp_data = dataset[temp_col].dropna()
        if not temp_data.empty:
             temp_bins = pd.cut(dataset[temp_col], bins=3, labels=['Low', 'Med', 'High'], duplicates='drop')
        else:
             temp_bins = pd.Series(['Med'] * len(dataset), index=dataset.index)
    else:
        temp_bins = pd.Series(['Med'] * len(dataset), index=dataset.index)

# DEFINE MANUAL CLUSTERS BASED ON SEASON AND TEMPERATURE BIN
dataset['cluster'] = dataset['season'].astype(str) + '_' + temp_bins.astype(str)

# DISPLAY UNIQUE CLUSTERS AND THEIR SIZES
cluster_summary = dataset.groupby('cluster').size()
print("MANUAL CLUSTERS CREATED:")
print(cluster_summary)

# SAVE CLUSTERED DATASET TO RESULTS FOLDER
clustered_path = '../results/clustered_dataset.csv'
try:
    # USAMOS ISO-8859-1 (latin1) para el guardado porque es el más compatible con entornos charmap/Windows
    dataset.to_csv(clustered_path, index=False, encoding='ISO-8859-1') 
    
    # VERIFY SAVE
    if os.path.exists(clustered_path):
        print(f"CONFIRMED: {clustered_path} SAVED SUCCESSFULLY.")
    else:
        # Esta parte solo se ejecuta si la función to_csv no lanzó una excepción, pero el archivo no existe.
        print(f"ERROR: {clustered_path} NOT SAVED (Possible blocking issue).")

except Exception as e:
    print(f"FATAL ERROR while saving {clustered_path}: {e}")
    traceback.print_exc()
    raise

print("CLUSTERING COMPLETED AND SAVED TO '../results/'.")