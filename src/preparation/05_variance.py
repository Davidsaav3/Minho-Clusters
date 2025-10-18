import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import json

# CONFIGURACIÓN
INPUT_CSV = '../../results/preparation/04_scale.csv'        # DATASET DE ENTRADA
OUTPUT_CSV = '../../results/preparation/05_variance.csv'    # DATASET FINAL
OUTPUT_JSON = '../../results/preparation/05_aux.json'       # COLUMNAS EXCLUIDAS
VAR_THRESHOLD = 0.01            # UMBRAL DE VARIANZA (ELIMINA COLUMNAS CON VAR < UMBRAL)
DROP_CONSTANT_COLUMNS = True     # ELIMINA COLUMNAS CON VAR = 0
DROP_LOW_VARIANCE = True         # ELIMINA COLUMNAS CON VAR < VAR_THRESHOLD
SAVE_SELECTED_COLUMNS = True     # GUARDAR COLUMNAS SELECCIONADAS
SAVE_EXCLUDED_COLUMNS = True     # GUARDAR COLUMNAS ELIMINADAS
SCALE_BEFORE = False             # ESCALAR COLUMNAS ANTES DE CALCULAR VARIANZA
NUMERIC_ONLY = True              # SOLO USAR COLUMNAS NUMÉRICAS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)
if NUMERIC_ONLY:
    df = df.select_dtypes(include=['int64','float64'])
print(f"[ INFO ] Dataset cargado: {df.shape}")

# ESCALADO OPCIONAL
if SCALE_BEFORE:
    df[df.columns] = StandardScaler().fit_transform(df)
    print("[ INFO ] Columnas escaladas antes de varianza")

# SELECCIÓN POR VARIANZA
if DROP_CONSTANT_COLUMNS or DROP_LOW_VARIANCE:
    selector = VarianceThreshold(threshold=VAR_THRESHOLD)
    df_selected = pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support()])
    excluded_cols = list(df.columns[~selector.get_support()])
    print(f"[ INFO ] Columnas seleccionadas: {df_selected.shape[1]} / {df.shape[1]}")
    print(f"[ INFO ] Columnas excluidas: {len(excluded_cols)}")
else:
    df_selected = df.copy()
    excluded_cols = []

# GUARDAR RESULTADOS
df_selected.to_csv(OUTPUT_CSV, index=False)
if SAVE_SELECTED_COLUMNS:
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({'selected_columns': df_selected.columns.tolist()}, f, indent=4)
if SAVE_EXCLUDED_COLUMNS:
    with open(OUTPUT_JSON.replace('.json','_excluded.json'), 'w') as f:
        json.dump({'excluded_columns': excluded_cols}, f, indent=4)

print("[ GUARDADO ] Dataset y columnas seleccionadas/excluidas")
