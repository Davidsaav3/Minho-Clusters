# src/01_exploracion.py
import pandas as pd

# Cargar dataset
df = pd.read_csv('../../data/dataset.csv')

# Resumen básico
info = df.describe(include='all').transpose()
info['nulos'] = df.isnull().sum()
info.to_csv('../../results/preparation/01_exploracion.csv')

print("Resumen inicial guardado en 'dataset_info.csv'")
