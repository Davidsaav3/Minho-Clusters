# FILE: continuity_analysis.py
# THIS SCRIPT ANALYZES THE CONTINUITY OF ANOMALIES BY ASSIGNING A SEQUENCE ORDER (1,2,3,...) AND 0 FOR NO ANOMALY TO EVALUATE PERSISTENCE AND TEMPORAL COHERENCE.

import pandas as pd
import numpy as np
import os

# CHECK IF DATASET WITH GLOBAL ANOMALIES EXISTS
global_path = '../results/dataset_with_global_anomalies.csv'
if not os.path.exists(global_path):
    raise FileNotFoundError(f"{global_path} NOT FOUND. RUN ISOLATION_FOREST_GLOBAL FIRST.")

print(f"LOADING DATASET FROM: {global_path}")

# LOAD DATASET WITH GLOBAL ANOMALIES FROM RESULTS FOLDER WITHOUT PARSE_DATES; ADD LOW_MEMORY=FALSE; USE ENCODING='UTF-8' WITH ERRORS='REPLACE'
with open(global_path, 'r', encoding='utf-8', errors='replace') as f:
    dataset = pd.read_csv(f, low_memory=False)

# RENAME COLUMN IF IT HAS BOM PREFIX (IN CASE IT WASN'T SAVED PROPERLY)
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': 'datetime'}, inplace=True)
    print("RENAMED 'ï»¿datetime' TO 'datetime'.")

# CONVERT 'DATETIME' COLUMN TO DATETIME AFTER LOADING IF IT EXISTS
if 'datetime' in dataset.columns:
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], errors='coerce')

# ASSUME 'DATETIME' COLUMN EXISTS AND IS SORTED; SORT DATASET BY DATETIME FOR TEMPORAL ANALYSIS
if 'datetime' in dataset.columns:
    dataset = dataset.sort_values('datetime')
else:
    print("WARNING: 'datetime' COLUMN NOT FOUND; SKIPPING SORT.")
    dataset = dataset.sort_index()  # FALLBACK TO INDEX SORT

# REGISTER SEQUENCE OF ANOMALIES: ASSIGN INCREMENTING ORDER (1,2,3,...) FOR CONSECUTIVE ANOMALIES, 0 OTHERWISE
anomaly_sequence = []
sequence_counter = 0
for label in dataset['global_anomaly']:
    if label == -1:  # ANOMALY DETECTED
        sequence_counter += 1
        anomaly_sequence.append(sequence_counter)
    else:
        anomaly_sequence.append(0)

# ADD SEQUENCE TO DATASET
dataset['anomaly_sequence'] = anomaly_sequence

# EVALUATE PERSISTENCE: COUNT CONSECUTIVE ANOMALIES (SEQUENCE > 0 AND PREVIOUS > 0)
dataset['is_persistent'] = (dataset['anomaly_sequence'] > 0) & (dataset['anomaly_sequence'].shift(1) > 0)
persistence_count = dataset['is_persistent'].sum()
print(f"PERSISTENT ANOMALY SEQUENCES: {persistence_count}")

# PRINT TEMPORAL COHERENCE SUMMARY: GROUP BY SEQUENCE COUNTER AND COUNT OCCURRENCES
coherence_summary = dataset[dataset['anomaly_sequence'] > 0].groupby('anomaly_sequence').size()
print("TEMPORAL COHERENCE SUMMARY (ANOMALY CLUSTER SIZES):")
print(coherence_summary.head())

# SAVE UPDATED DATASET TO RESULTS FOLDER; SPECIFY ENCODING='UTF-8'
continuity_path = '../results/dataset_with_continuity.csv'
dataset.to_csv(continuity_path, index=False, encoding='utf-8')

# VERIFY SAVE
if os.path.exists(continuity_path):
    print("CONFIRMED: dataset_with_continuity.csv SAVED SUCCESSFULLY.")
else:
    print("ERROR: dataset_with_continuity.csv NOT SAVED.")

print("CONTINUITY ANALYSIS COMPLETED AND SAVED TO '../results/'.")