import pandas as pd  # MANEJO DE DATAFRAMES
import os            # MANEJO DE RUTAS Y DIRECTORIOS

# PARÁMETROS CONFIGURABLES
INPUT_FILE = '../../data/dataset.csv'                # DATASET DE ENTRADA
OUTPUT_FILE = '../../results/preparation/01_metrics.csv'  # CSV CON RESUMEN
INCLUDE_DESCRIBE = 'all'                             # 'all', 'number', 'object'
SHOW_INFO = True                                     # TRUE = MOSTRAR INFO EN PANTALLA
CREATE_DIR_IF_MISSING = True                          # TRUE = CREAR CARPETA SI NO EXISTE

# CREAR CARPETA RESULTADOS SI NO EXISTE
output_dir = os.path.dirname(OUTPUT_FILE)
if CREATE_DIR_IF_MISSING:
    os.makedirs(output_dir, exist_ok=True)
    if SHOW_INFO:
        print(f"[ INFO ] Carpeta '{output_dir}' creada si no existía")

# CARGAR DATASET
df = pd.read_csv(INPUT_FILE)
if SHOW_INFO:
    print(f"[ INFO ] Dataset cargado: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# RESUMEN ESTADÍSTICO
info = df.describe(include=INCLUDE_DESCRIBE).transpose()  # ESTADÍSTICAS DESCRIPTIVAS
info['nulos'] = df.isnull().sum()                          # CONTAR VALORES NULOS
if SHOW_INFO:
    print("[ INFO ] Estadísticas descriptivas calculadas")

# GUARDAR RESULTADO
info.to_csv(OUTPUT_FILE)
if SHOW_INFO:
    print(f"[ GUARDADO ] Resumen inicial guardado en '{OUTPUT_FILE}'")
