# FILE: isolation_forest_per_cluster.py
# THIS SCRIPT APPLIES ISOLATION FOREST TO EACH CLUSTER, COMPARES WITH GLOBAL RESULTS, AND IDENTIFIES REPRESENTATIVE ANOMALIES.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import defaultdict
import os
import traceback

# CHECK IF CLUSTERED DATASET EXISTS
clustered_path = '../results/clustered_dataset.csv'
if not os.path.exists(clustered_path):
    raise FileNotFoundError(f"{clustered_path} NOT FOUND. RUN CLUSTERING FIRST.")

print(f"LOADING DATASET FROM: {clustered_path}")

try:
    print("Attempting to load CSV with encoding 'latin1'...")
    dataset = pd.read_csv(clustered_path, low_memory=False, encoding='latin1')
    print("CSV loaded successfully with latin1.")
except Exception as e:
    print(f"Error loading with latin1: {e}")
    print("Trying with 'utf-8' and errors='replace'...")
    try:
        with open(clustered_path, 'r', encoding='utf-8', errors='replace') as f:
            dataset = pd.read_csv(f, low_memory=False)
        print("CSV loaded successfully with utf-8 and errors='replace'.")
    except Exception as e2:
        print(f"Error loading with utf-8: {e2}")
        print("Full traceback:")
        traceback.print_exc()
        raise

# RENAME COLUMN IF IT HAS BOM PREFIX (IN CASE IT WASN'T SAVED PROPERLY)
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': 'datetime'}, inplace=True)
    print("RENAMED 'ï»¿datetime' TO 'datetime'.")

# CONVERT 'DATETIME' COLUMN TO DATETIME AFTER LOADING IF IT EXISTS, WITH ERRORS='IGNORE' TO SKIP BAD VALUES
if 'datetime' in dataset.columns:
    print("Converting datetime column...")
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], errors='ignore')
    print("Datetime conversion completed.")
else:
    print("WARNING: 'datetime' column not found.")

print(f"Dataset shape after loading: {dataset.shape}")
print(f"Columns: {dataset.columns.tolist()[:5]}...")  # Print first 5 columns for debug

# SELECT FEATURE COLUMNS
temporal_cols = ['datetime', 'date', 'time', 'year', 'month', 'day', 'hour', 'minute', 'weekday', 'day_of_year', 'week_of_year', 'working_day', 'season', 'holiday', 'weekend', 'global_anomaly', 'anomaly_sequence', 'is_persistent', 'cluster']
feature_columns = dataset.select_dtypes(include=[np.number]).columns.drop(temporal_cols, errors='ignore').tolist()
print(f"Selected feature columns: {len(feature_columns)} columns")

# GROUP DATASET BY CLUSTER FOR PER-CLUSTER ANALYSIS
print("Grouping by cluster...")
clusters = dataset.groupby('cluster')
print(f"Number of clusters: {len(clusters)}")

# DICTIONARY TO STORE PER-CLUSTER ANOMALIES
cluster_anomalies = defaultdict(list)
coincident_anomalies = []

# ITERATE OVER EACH CLUSTER
print("Processing clusters...")
for i, (cluster_name, cluster_data) in enumerate(clusters):
    print(f"Processing cluster {i+1}/{len(clusters)}: {cluster_name} (size: {len(cluster_data)})")
    if len(cluster_data) < 10:  # SKIP SMALL CLUSTERS
        print(f"Skipping small cluster {cluster_name}")
        continue
    
    try:
        X_cluster = cluster_data[feature_columns]
        print(f"X_cluster shape: {X_cluster.shape}")
        
        # APPLY ISOLATION FOREST PER CLUSTER WITH SAME CONTAMINATION
        contamination_level = 0.05
        cluster_if = IsolationForest(contamination=contamination_level, random_state=42)
        cluster_labels = cluster_if.fit_predict(X_cluster)
        print(f"Cluster {cluster_name}: Anomalies detected: {np.sum(cluster_labels == -1)}")
        
        # STORE CLUSTER ANOMALY INDICES (RELATIVE TO FULL DATASET)
        anomaly_indices = cluster_data.index[cluster_labels == -1].tolist()
        cluster_anomalies[cluster_name] = anomaly_indices
        
        # COMPARE WITH GLOBAL: FIND COINCIDENT ANOMALIES (WHERE CLUSTER LABEL == -1 AND GLOBAL == -1)
        global_anoms_in_cluster = cluster_data[cluster_data['global_anomaly'] == -1].index.tolist()
        coincident = set(anomaly_indices) & set(global_anoms_in_cluster)
        coincident_anomalies.extend(list(coincident))
        print(f"Coincident anomalies in {cluster_name}: {len(coincident)}")
    except Exception as e:
        print(f"Error processing cluster {cluster_name}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        continue

# COUNT TOTAL COINCIDENT ANOMALIES (MOST REPRESENTATIVE)
print(f"TOTAL COINCIDENT ANOMALIES BETWEEN GLOBAL AND CLUSTERS: {len(coincident_anomalies)}")

# LOCALIZE ORIGIN: ANALYZE CONCENTRATION BY SEASON OR CLUSTER
print("Analyzing coincident anomalies...")
coincident_df = dataset.loc[coincident_anomalies]
concentration_by_season = coincident_df['season'].value_counts()
print("ANOMALY CONCENTRATION BY SEASON:")
print(concentration_by_season.head())

concentration_by_cluster = coincident_df['cluster'].value_counts()
print("ANOMALY CONCENTRATION BY CLUSTER:")
print(concentration_by_cluster.head())

# CHARACTERIZE AND EXPLAIN ANOMALIES BY CLUSTER ANALYSIS
# EXAMPLE: FOR HIGH CONCENTRATION IN A SEASON ON HOT DAYS
for season, count in concentration_by_season.head(1).items():
    if 'aemet_temperatura_media' in coincident_df.columns:
        hot_days_anoms = coincident_df[(coincident_df['season'] == season) & (coincident_df['aemet_temperatura_media'] > coincident_df['aemet_temperatura_media'].quantile(0.9))]
        if len(hot_days_anoms) > 0:
            print(f"EJEMPLO: ESTA ANOMALÍA SE PRODUCE EN LA ESTACIÓN {season} Y APARECE PRINCIPALMENTE EN DÍAS DE CALOR ({len(hot_days_anoms)} INSTANCIAS).")
        else:
            print(f"EJEMPLO: ESTA ANOMALÍA SE PRODUCE EN LA ESTACIÓN {season} ({count} INSTANCIAS).")
    else:
        print(f"EJEMPLO: ESTA ANOMALÍA SE PRODUCE EN LA ESTACIÓN {season} ({count} INSTANCIAS).")

# SAVE FINAL DATASET WITH CLUSTER LABELS (SIMPLIFIED) TO RESULTS FOLDER; SPECIFY ENCODING='UTF-8'
print("Saving final dataset...")
final_path = '../results/final_analysis_dataset.csv'
cluster_labels_full = np.ones(len(dataset))
for indices in cluster_anomalies.values():
    cluster_labels_full[indices] = -1  # OVERLAP; ACTUAL PER-CLUSTER VARY
dataset['cluster_anomaly'] = cluster_labels_full
dataset.to_csv(final_path, index=False, encoding='utf-8')

# VERIFY SAVE
if os.path.exists(final_path):
    print("CONFIRMED: final_analysis_dataset.csv SAVED SUCCESSFULLY.")
else:
    print("ERROR: final_analysis_dataset.csv NOT SAVED.")

print("PER-CLUSTER ISOLATION FOREST AND COMPARISON COMPLETED; SAVED TO '../results/'.")