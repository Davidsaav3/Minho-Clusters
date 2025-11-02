import pandas as pd
import numpy as np
import os

# ================= CONFIGURACIÓN =================
INPUT_CSV = '../../results/preparation/05_variance_recortado.csv'
OUTPUT_FOLDER = '../../results/execution/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

COLUMNS_TO_NOT_CONTAMINATE = ["datetime","date","time","year","month","day","hour","minute","weekday",
                              "day_of_year","week_of_year","working_day","season","holiday","weekend",
                              "aemet_temperatura_media","aemet_precipitaciones","aemet_temperatura_minima",
                              "aemet_maxima_temperatura","aemet_direccion_media_viento","aemet_velocidad_media_viento",
                              "aemet_racha_maxima","aemet_humedad_media","aemet_humediad_maxima","aemet_humedad_minima",
                              "openweather_temperatura_media","openweather_punto_rocio","openweather_temperatura_sensacion",
                              "openweather_temperatura_minima","openweather_temperatura_maxima","openweather_presion_atmosferica",
                              "openweather_humedad_media","openweather_velocidad_viento_media","openweather_direccion_viento_media",
                              "openweather_racha_maxima_viento","openweather_lluvia_1h","openweather_cobertura_nubosa"]

OPERACION_RUIDO = 'SUMA'
ADD_LABEL = True
RANDOM_STATE = 42
SHOW_INFO = True

# ================= PARÁMETROS VARIADOS =================
contamination_rates = [0.01, 0.005, 0.001, 0.0]  # del 1% hasta 0%
noise_intensities = [3.0, 2.0, 1.0, 0.0]       # desde el valor actual hasta 0

# ================= CARGA DEL DATASET =================
df = pd.read_csv(INPUT_CSV)
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} filas, {df.shape[1]} columnas")

# Columnas numéricas a contaminar
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cols_to_contaminate = [c for c in numeric_cols if c not in COLUMNS_TO_NOT_CONTAMINATE]

if not cols_to_contaminate:
    raise ValueError("[ ERROR ] No hay columnas válidas para contaminar")

# ================= LOOP PARA GENERAR VARIANTES =================
for cr in contamination_rates:
    for ni in noise_intensities:
        np.random.seed(RANDOM_STATE)
        df_contaminated = df.copy()
        n_rows = df.shape[0]
        n_contam = int(n_rows * cr)
        if n_contam > 0:
            contam_indices = np.random.choice(df.index, size=n_contam, replace=False)
        else:
            contam_indices = []

        # Aplicar ruido a las filas seleccionadas
        for col in cols_to_contaminate:
            if df[col].dtype in ['int64', 'float64'] and n_contam > 0:
                noise = np.random.normal(0, ni * df[col].std(), size=n_contam)
                if OPERACION_RUIDO == 'SUMA':
                    df_contaminated.loc[contam_indices, col] += noise
                elif OPERACION_RUIDO == 'RESTA':
                    df_contaminated.loc[contam_indices, col] -= noise
                elif OPERACION_RUIDO == 'MULTIPLICACION':
                    df_contaminated.loc[contam_indices, col] *= (1 + noise)
                elif OPERACION_RUIDO == 'ESCALA':
                    df_contaminated.loc[contam_indices, col] = df[col].mean() + noise

        # Añadir columna de anomalía
        if ADD_LABEL:
            df_contaminated['is_anomaly'] = 0
            if n_contam > 0:
                df_contaminated.loc[contam_indices, 'is_anomaly'] = 1

        # Nombre de archivo indicando configuración
        output_name = f"contaminated_cr{cr}_ni{ni}.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        df_contaminated.to_csv(output_path, index=False)
        if SHOW_INFO:
            print(f"[ GUARDADO ] {output_name} -> Filas contaminadas: {n_contam}, Noise: {ni}")
