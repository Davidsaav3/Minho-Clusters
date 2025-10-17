# src/04_escalado.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../../results/preparation/03_codificacion.csv')

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df_scaled.to_csv('../../results/preparation/04_escalado.csv', index=False)
print("Dataset escalado guardado en '../../results/preparation/04_escalado.csv'")
