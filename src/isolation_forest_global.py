# FILE: isolation_forest_global.py
# THIS SCRIPT APPLIES ISOLATION FOREST TO THE FULL DATASET FOR A GLOBAL ANOMALY DETECTION.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

# CHECK IF CONTAMINATED DATASET EXISTS
contaminated_path = '../results/contaminated_dataset.csv'
if not os.path.exists(contaminated_path):
    raise FileNotFoundError(f"{contaminated_path} NOT FOUND. RUN PRELIMINARY_ANALYSIS FIRST.")

print(f"LOADING DATASET FROM: {contaminated_path}")

# LOAD THE CONTAMINATED DATASET FOR GLOBAL ANALYSIS FROM RESULTS FOLDER WITHOUT PARSE_DATES; ADD LOW_MEMORY=FALSE; USE ENCODING='UTF-8' WITH ERRORS='REPLACE'
with open(contaminated_path, 'r', encoding='utf-8', errors='replace') as f:
    dataset = pd.read_csv(f, low_memory=False)

# RENAME COLUMN IF IT HAS BOM PREFIX (IN CASE IT WASN'T SAVED PROPERLY)
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': 'datetime'}, inplace=True)
    print("RENAMED 'ï»¿datetime' TO 'datetime'.")

# CONVERT 'DATETIME' COLUMN TO DATETIME AFTER LOADING IF IT EXISTS
if 'datetime' in dataset.columns:
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], errors='coerce')

# SELECT FEATURE COLUMNS AS IN PRELIMINARY ANALYSIS; RELOAD TO ENSURE CONSISTENCY
temporal_cols = ['datetime', 'date', 'time', 'year', 'month', 'day', 'hour', 'minute', 'weekday', 'day_of_year', 'week_of_year', 'working_day', 'season', 'holiday', 'weekend']
feature_columns = dataset.select_dtypes(include=[np.number]).columns.drop(temporal_cols, errors='ignore').tolist()
X = dataset[feature_columns]

# APPLY ISOLATION FOREST TO THE FULL DATASET WITH ADJUSTED CONTAMINATION LEVEL FOR GLOBAL DETECTION
contamination_level = 0.05
global_if_model = IsolationForest(contamination=contamination_level, random_state=42)
global_anomaly_labels = global_if_model.fit_predict(X)

# ADD ANOMALY LABELS TO DATASET; -1 FOR ANOMALY, 1 FOR NORMAL
dataset['global_anomaly'] = global_anomaly_labels

# COUNT GLOBAL ANOMALIES
global_anomalies_count = np.sum(global_anomaly_labels == -1)
print(f"GLOBAL ANOMALIES DETECTED: {global_anomalies_count}")

# SAVE DATASET WITH GLOBAL ANOMALY LABELS TO RESULTS FOLDER; SPECIFY ENCODING='UTF-8'
global_path = '../results/dataset_with_global_anomalies.csv'
dataset.to_csv(global_path, index=False, encoding='utf-8')

# VERIFY SAVE
if os.path.exists(global_path):
    print("CONFIRMED: dataset_with_global_anomalies.csv SAVED SUCCESSFULLY.")
else:
    print("ERROR: dataset_with_global_anomalies.csv NOT SAVED.")

print("GLOBAL ISOLATION FOREST EXECUTED AND SAVED TO '../results/'.")