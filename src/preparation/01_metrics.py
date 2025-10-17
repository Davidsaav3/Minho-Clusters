# IMPORTS
import pandas as pd  # MANEJO DE DATAFRAMES

# CARGAR DATASET
df = pd.read_csv('../../data/dataset.csv')
print(f"[ INFO ] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# RESUMEN BÁSICO
info = df.describe(include='all').transpose()  # ESTADÍSTICAS DESCRIPTIVAS
info['nulos'] = df.isnull().sum()               # CONTAR VALORES NULOS

# GUARDAR RESULTADO
info.to_csv('../../results/preparation/01_metrics.csv')
print("[ GUARDADO ] Resumen inicial en '../../results/preparation/01_metrics.csv'")
