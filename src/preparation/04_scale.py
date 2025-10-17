# IMPORTS
import pandas as pd  # MANEJO DE DATAFRAMES
from sklearn.preprocessing import StandardScaler  # ESCALADO DE DATOS

# CARGAR DATASET CODIFICADO
df = pd.read_csv('../../results/preparation/03_codification.csv')
print(f"[ INFO ] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# ESCALAR DATOS (MEDIA=0, STD=1)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("[ INFO ] Dataset escalado con StandardScaler (media=0, std=1)")

# GUARDAR DATASET ESCALADO
df_scaled.to_csv('../../results/preparation/04_scale.csv', index=False)
print("[ GUARDADO ] Dataset escalado en '../../results/preparation/04_scale.csv'")
