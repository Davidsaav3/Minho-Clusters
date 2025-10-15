# FILE: data_preparation.py
# THIS SCRIPT HANDLES THE PREPARATION OF THE DATASET BY REMOVING ROWS WITH NULL VALUES IN NUMERIC COLUMNS TO ENSURE A CLEAN DATASET FOR ANALYSIS.

import pandas as pd
import numpy as np
import os

# CREATE RESULTS DIRECTORY IF IT DOESN'T EXIST
os.makedirs('../results', exist_ok=True)

# LOAD THE DATASET FROM THE SPECIFIED PATH WITHOUT PARSE_DATES; USE ENCODING='LATIN1' TO HANDLE SPECIAL CHARACTERS
dataset = pd.read_csv('../data/dataset.csv', low_memory=False, encoding='latin1')

# RENAME COLUMN IF IT HAS BOM PREFIX
if 'ï»¿datetime' in dataset.columns:
    dataset.rename(columns={'ï»¿datetime': 'datetime'}, inplace=True)
    print("RENAMED 'ï»¿datetime' TO 'datetime'.")

# CONVERT 'DATETIME' COLUMN TO DATETIME AFTER LOADING
if 'datetime' in dataset.columns:
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], errors='coerce')
else:
    print("WARNING: 'datetime' COLUMN NOT FOUND. AVAILABLE COLUMNS:")
    print(dataset.columns.tolist()[:10])  # PRINT FIRST 10 COLUMNS

# DISPLAY INITIAL SHAPE TO UNDERSTAND THE DATASET SIZE BEFORE CLEANING
print(f"INITIAL DATASET SHAPE: {dataset.shape}")

# CHECK FOR NULL VALUES IN EACH COLUMN TO IDENTIFY MISSING DATA
null_counts = dataset.isnull().sum()
print("NULL COUNTS PER COLUMN:")
print(null_counts[null_counts > 0])  # SHOW ONLY COLUMNS WITH NULLS

# DEFINE TEMPORAL/CATEGORICAL COLUMNS TO EXCLUDE FROM NULL CHECK
temporal_cols = ['datetime', 'date', 'time', 'year', 'month', 'day', 'hour', 'minute', 'weekday', 'day_of_year', 'week_of_year', 'working_day', 'season', 'holiday', 'weekend']

# SELECT NUMERICAL COLUMNS FOR NULL REMOVAL
numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
print(f"NUMERIC COLUMNS SELECTED FOR NULL REMOVAL: {len(numeric_cols)} columns")

# REMOVE ROWS WITH NULL VALUES ONLY IN NUMERIC COLUMNS
clean_dataset = dataset.dropna(subset=numeric_cols)

# DISPLAY SHAPE AFTER CLEANING TO CONFIRM ROWS REMOVED
print(f"CLEAN DATASET SHAPE: {clean_dataset.shape}")

# SAVE THE CLEAN DATASET TO A NEW CSV FILE IN RESULTS FOLDER; SPECIFY ENCODING='UTF-8'
clean_path = '../results/clean_dataset.csv'
clean_dataset.to_csv(clean_path, index=False, encoding='utf-8')

# VERIFY SAVE
if os.path.exists(clean_path):
    print("CONFIRMED: clean_dataset.csv SAVED SUCCESSFULLY.")
else:
    print("ERROR: clean_dataset.csv NOT SAVED.")

# PRINT A CONFIRMATION MESSAGE
print("DATASET CLEANING COMPLETED AND SAVED AS '../results/clean_dataset.csv'")