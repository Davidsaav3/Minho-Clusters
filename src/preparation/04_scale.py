import pandas as pd  # MANEJO DE DATAFRAMES
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# PARÁMETROS CONFIGURABLES
INPUT_FILE = '../../results/preparation/03_codification.csv'  # DATASET DE ENTRADA
OUTPUT_FILE = '../../results/preparation/04_scale.csv'        # DATASET ESCALADO GUARDADO
SCALER_TYPE = 'standard'                                      # TIPO DE ESCALADO: 'standard', 'minmax', 'robust', 'maxabs'
SHOW_INFO = True                                              # TRUE = MOSTRAR INFO EN PANTALLA
SAVE_INTERMEDIATE = False                                     # TRUE = GUARDAR DATASET INTERMEDIO ANTES DE ESCALAR
FEATURES_TO_SCALE = None                                       # LISTA DE COLUMNAS A ESCALAR, NONE = TODAS
CLIP_VALUES = False                                           # TRUE = RECORTAR VALORES EXTREMOS DESPUÉS DE ESCALADO
CLIP_MIN = 0                                                  # MÍNIMO AL RECORTAR
CLIP_MAX = 1                                                  # MÁXIMO AL RECORTAR

# CARGAR DATASET
df = pd.read_csv(INPUT_FILE)
if SHOW_INFO:
    print(f"[ INFO ] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# SELECCIONAR COLUMNAS A ESCALAR
if FEATURES_TO_SCALE is None:
    FEATURES_TO_SCALE = df.columns.tolist()
if SHOW_INFO:
    print(f"[ INFO ] Columnas a escalar: {len(FEATURES_TO_SCALE)}")

# SELECCIÓN DE ESCALADOR
if SCALER_TYPE == 'standard':
    scaler = StandardScaler()  
    # RECOMENDADO PARA DISTRIBUCIÓN NORMAL (Ej: KMeans, PCA, LSTM)
elif SCALER_TYPE == 'minmax':
    scaler = MinMaxScaler()    
    # RECOMENDADO CUANDO SE NORMALIZA ENTRE 0 Y 1, ÚTIL PARA REDES NEURONALES
elif SCALER_TYPE == 'robust':
    scaler = RobustScaler()    
    # RECOMENDADO SI HAY OUTLIERS FUERTES
elif SCALER_TYPE == 'maxabs':
    scaler = MaxAbsScaler()    
    # RECOMENDADO PARA DATOS QUE PUEDEN SER NEGATIVOS, ESCALA ENTRE -1 Y 1
else:
    raise ValueError(f"[ ERROR ] Escalador desconocido: {SCALER_TYPE}")

# GUARDAR DATASET INTERMEDIO (OPCIONAL)
if SAVE_INTERMEDIATE:
    intermediate_file = OUTPUT_FILE.replace('.csv','_intermediate.csv')
    df.to_csv(intermediate_file, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] Dataset intermedio en '{intermediate_file}'")

# ESCALAR DATOS
df_scaled = df.copy()
df_scaled[FEATURES_TO_SCALE] = scaler.fit_transform(df_scaled[FEATURES_TO_SCALE])
if SHOW_INFO:
    print(f"[ INFO ] Dataset escalado usando '{SCALER_TYPE}'")

# RECORTAR VALORES EXTREMOS (OPCIONAL)
if CLIP_VALUES:
    df_scaled[FEATURES_TO_SCALE] = df_scaled[FEATURES_TO_SCALE].clip(CLIP_MIN, CLIP_MAX)
    if SHOW_INFO:
        print(f"[ INFO ] Valores recortados entre {CLIP_MIN} y {CLIP_MAX}")

# GUARDAR DATASET ESCALADO
df_scaled.to_csv(OUTPUT_FILE, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] Dataset escalado en '{OUTPUT_FILE}'")
