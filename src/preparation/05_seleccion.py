# src/05_seleccion.py
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

df = pd.read_csv('../../results/preparation/04_escalado.csv')

# Eliminar columnas con varianza cero
selector = VarianceThreshold(threshold=0.0)
df_selected = pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support()])

df_selected.to_csv('../../results/preparation/05_seleccion.csv', index=False)
print("Dataset final guardado en '../../results/preparation/05_seleccion.csv'")
