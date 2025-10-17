# IMPORTS
import pandas as pd  # MANEJO DE DATAFRAMES
from sklearn.feature_selection import VarianceThreshold  # SELECCIÓN POR VARIANZA
import json  # GUARDAR COLUMNAS EXCLUIDAS

# CARGAR DATASET ESCALADO
df = pd.read_csv('../../results/preparation/04_scale.csv')
print(f"[ INFO ] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# ELIMINAR COLUMNAS CON VARIANZA CERO
selector = VarianceThreshold(threshold=0.0)
df_selected = pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support()])
excluded_cols = list(df.columns[~selector.get_support()])
print(f"[ INFO ] Columnas seleccionadas: {df_selected.shape[1]} de {df.shape[1]} (varianza > 0)")
print(f"[ INFO ] Columnas excluidas por varianza cero")

# GUARDAR DATASET FINAL
df_selected.to_csv('../../results/preparation/05_variance.csv', index=False)
print("[ GUARDADO ] Dataset final en '../../results/preparation/05_variance.csv'")

# GUARDAR COLUMNAS EXCLUIDAS EN JSON
with open('../../results/preparation/06_aux.json', 'w') as f:
    json.dump({'excluded_columns': excluded_cols}, f, indent=4)
print("[ GUARDADO ] Columnas excluidas guardadas en '../../results/preparation/06_aux.json'")
