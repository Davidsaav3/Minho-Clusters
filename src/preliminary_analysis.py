# FILE: preliminary_analysis.py
# THIS SCRIPT PERFORMS A PRELIMINARY ANALYSIS TO VERIFY CLEANLINESS, TEST ISOLATION FOREST WITH INITIAL CONTAMINATION, INTRODUCE ARTIFICIAL CONTAMINATION, AND PROVIDE STATISTICAL SUMMARIES.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings
import os
warnings.filterwarnings("ignore")

# CHECK IF CLEAN DATASET EXISTS
clean_path = '../results/clean_dataset.csv'
if not os.path.exists(clean_path):
    raise FileNotFoundError(f"{clean_path} NOT FOUND. RUN DATA_PREPARATION FIRST.")

print(f"LOADING DATASET FROM: {clean_path}")

# LOAD THE CLEAN DATASET PREPARED IN THE PREVIOUS STEP FROM RESULTS FOLDER WITHOUT PARSE_DATES; ADD LOW_MEMORY=FALSE; USE ENCODING='UTF-8' WITH ERRORS='REPLACE'
with open(clean_path, 'r', encoding='utf-8', errors='replace') as f:
    dataset = pd.read_csv(f, low_memory=False)

# RENAME COLUMN IF IT HAS BOM PREFIX (IN CASE IT WASN'T SAVED PROPERLY)
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': 'datetime'}, inplace=True)
    print("RENAMED 'ï»¿datetime' TO 'datetime'.")

# CONVERT 'DATETIME' COLUMN TO DATETIME AFTER LOADING IF IT EXISTS
if 'datetime' in dataset.columns:
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], errors='coerce')

# VERIFY CLEANLINESS BY CHECKING FOR ANY REMAINING NULL VALUES
if dataset.isnull().sum().sum() == 0:
    print("DATASET IS CONFIRMED CLEAN: NO NULL VALUES FOUND.")
else:
    print("WARNING: DATASET STILL CONTAINS NULL VALUES.")

# DEFINE TEMPORAL/CATEGORICAL COLUMNS TO EXCLUDE FROM FEATURES
temporal_cols = ['datetime', 'date', 'time', 'year', 'month', 'day', 'hour', 'minute', 'weekday', 'day_of_year', 'week_of_year', 'working_day', 'season', 'holiday', 'weekend']

# SELECT NUMERICAL FEATURE COLUMNS FOR ISOLATION FOREST; EXCLUDE TEMPORAL ONES
feature_columns = dataset.select_dtypes(include=[np.number]).columns.drop(temporal_cols, errors='ignore').tolist()
print(f"SELECTED FEATURE COLUMNS: {feature_columns[:10]}... (total: {len(feature_columns)})")

# DEFINE INITIAL CONTAMINATION LEVEL FOR ISOLATION FOREST; 0.01 MEANS APPROX 1% ANOMALIES (1 PER 1000 RECORDS)
contamination_level = 0.01

# PREPARE FEATURES FOR ISOLATION FOREST
X = dataset[feature_columns]

# INITIALIZE AND FIT ISOLATION FOREST MODEL WITH SPECIFIED CONTAMINATION
if_model = IsolationForest(contamination=contamination_level, random_state=42)
anomaly_labels = if_model.fit_predict(X)

# DETECT POSSIBLE NATURAL ANOMALIES; -1 INDICATES ANOMALY, COUNT THEM (EXPECTED ~1000 SUSPECTS)
natural_anomalies_count = np.sum(anomaly_labels == -1)
print(f"NATURAL ANOMALIES DETECTED: {natural_anomalies_count}")

# INTRODUCE ARTIFICIAL CONTAMINATION IN CONTROLLED POINTS; FOR EXAMPLE, ADD EXTREME VALUES TO 5% OF RANDOM ROWS IN A KEY FEATURE LIKE 'AEMET_TEMPERATURA_MEDIA'
np.random.seed(42)
artificial_indices = np.random.choice(X.index, size=int(0.05 * len(X)), replace=False)
X_artificial = X.copy()
if 'aemet_temperatura_media' in X_artificial.columns:
    X_artificial.loc[artificial_indices, 'aemet_temperatura_media'] += np.random.uniform(20, 50, size=len(artificial_indices))  # SIMULATE HEAT ANOMALIES
else:
    # FALLBACK TO FIRST NUMERICAL COLUMN
    first_col = feature_columns[0]
    X_artificial.loc[artificial_indices, first_col] += np.random.uniform(50, 100, size=len(artificial_indices))

# RE-FIT ISOLATION FOREST ON ARTIFICIALLY CONTAMINATED DATA; ADJUST TO 0.05 IF NEEDED TO AVOID HIDING REAL ANOMALIES
adjusted_contamination = 0.05
if_model_adjusted = IsolationForest(contamination=adjusted_contamination, random_state=42)
anomaly_labels_adjusted = if_model_adjusted.fit_predict(X_artificial)

# COUNT ADJUSTED ANOMALIES
adjusted_anomalies_count = np.sum(anomaly_labels_adjusted == -1)
print(f"ADJUSTED ANOMALIES DETECTED AFTER ARTIFICIAL CONTAMINATION: {adjusted_anomalies_count}")

# COMPUTE STATISTICAL SUMMARY (MAX, MIN, MEAN, MEDIAN) FOR EACH FEATURE TO ANALYZE DISTRIBUTION; SHOW FOR FIRST 10 TO AVOID OVERLOAD
stats_summary = X[feature_columns[:10]].describe().loc[['min', 'max', 'mean', '50%']]  # 50% IS MEDIAN
print("STATISTICAL SUMMARY OF FIRST 10 FEATURES:")
print(stats_summary)

# EVALUATE CRITICAL COLUMNS; IDENTIFY THOSE WITH EXTREMES (E.G., MAX > 3*MEAN OR MIN < MEAN/3)
critical_columns = []
for col in feature_columns:
    if pd.notna(X[col].max()) and pd.notna(X[col].mean()) and (X[col].max() > 3 * X[col].mean() or X[col].min() < X[col].mean() / 3):
        critical_columns.append(col)

print(f"CRITICAL COLUMNS WITH EXTREME VALUES: {critical_columns[:5]}... (total: {len(critical_columns)})")

# SAVE THE ARTIFICIALLY CONTAMINATED DATASET TO RESULTS FOLDER FOR FURTHER USE; SPECIFY ENCODING='UTF-8'
contaminated_path = '../results/contaminated_dataset.csv'
X_artificial_df = dataset.copy()
X_artificial_df[feature_columns] = X_artificial
X_artificial_df.to_csv(contaminated_path, index=False, encoding='utf-8')

# VERIFY SAVE
if os.path.exists(contaminated_path):
    print("CONFIRMED: contaminated_dataset.csv SAVED SUCCESSFULLY.")
else:
    print("ERROR: contaminated_dataset.csv NOT SAVED.")

print("PRELIMINARY ANALYSIS COMPLETED; CONTAMINATED DATASET SAVED TO '../results/'.")