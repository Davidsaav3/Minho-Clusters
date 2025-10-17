# src/02_cluster_manual.py
import pandas as pd
import os

# =========================
# Cargar dataset
# =========================
df = pd.read_csv('../../results/preparation/05_seleccion.csv')

# Crear carpeta de resultados si no existe
os.makedirs('../results', exist_ok=True)

# =========================
# Definición de clusters y subparticiones
# =========================
clusters = {
    'temporal': {
        'Date_and_Time': ['datetime','date','time','year','month','day','hour','minute','weekday','day_of_year','week_of_year'],
        'Calendar': ['working_day','season','holiday','weekend']
    },
    'ubicacion': {
        'Pozo': ['nivel_plaxiquet','distancia_brocal_pozo_al_agua_falconera','distancia_brocal_pozo_al_agua'],
        'Falconera': ['presion_salida_falconera','caudal_entrada_falconera','cloro_falconera',
                      'intensidad_falconera_1','intensidad_falconera_2','intensidad_falconera_3',
                      'potencia_pozo_falconera','tension_variador_pozo_falconera','frecuencia_pozo_falconera','factor_potencia_falconera'],
        'Ull_Pueblo': ['presion_salida_ull_pueblo','caudal_salida_ull_pueblo','cloro_ull','nitratos_ull','ph_ull','temperatura_entrada_ull'],
        'Playa': ['presion_playa','caudal_instantaneo_playa','temperatura_playa','playa_horas_funcionamiento','playa_potencia','playa_tension_fase_1_2'],
        'Beniopa': ['beniopa_intensidad_a1','beniopa_intensidad_a2','beniopa_intensidad_a3','beniopa_factor_coste',
                    'beniopa_total_horas_funcionamiento','beniopa_caudal_pozo_impulsion','beniopa_potencia','beniopa_tension_variador'],
        'Llombart': ['llombart_intensidad_1_a','llombart_intensidad_2_a','llombart_intensidad_3_a','llombart_factor_potencia',
                     'llombart_total_horas_funcionamiento','llombart_caudal_impulsion','llombart_potencia'],
        'Sanjuan': ['sanjuan_intensidad_1_a','sanjuan_factor_potencia','sanjuan_caudal_impulsion','sanjuan_potencia']
    },
    'medida': {
        'Calidad_Agua': ['ph_falconera','ph_oficina','ph_playa','ph_ull',
                         'cloro_falconera','cloro_plaxiquet','cloro_oficina','cloro_ull',
                         'conductividad_falconera','conductividad_ull','nitratos_falconera','nitratos_ull'],
        'Hidraulica': ['presion_salida_falconera','presion_salida_ull_pueblo','presion_playa','caudal_instantaneo_playa',
                       'nivel_plaxiquet','distancia_brocal_pozo_al_agua','distancia_brocal_pozo_al_agua_falconera'],
        'Operacional': ['total_horas_funcionamiento_pozo_falconera','potencia_pozo_falconera',
                        'intensidad_falconera_1','intensidad_falconera_2','intensidad_falconera_3',
                        'factor_potencia_falconera','tension_variador_pozo_falconera','frecuencia_pozo_falconera']
    },
    'ambientales': {
        'Climatologia': ['aemet_temperatura_media','aemet_precipitaciones','aemet_temperatura_minima','aemet_maxima_temperatura',
                         'aemet_direccion_media_viento','aemet_velocidad_media_viento','aemet_racha_maxima','aemet_humedad_media',
                         'openweather_temperatura_media','openweather_punto_rocio','openweather_temperatura_minima','openweather_temperatura_maxima',
                         'openweather_presion_atmosferica','openweather_humedad_media','openweather_velocidad_viento_media','openweather_direccion_viento_media'],
        'Season_and_Calendar': ['season','holiday','weekend','working_day']
    },
    'eficiencia': {
        'Consumo_Electrico': ['potencia_pozo_falconera','potencia_variador_pozo_falconera','intensidad_falconera_1','intensidad_falconera_2','intensidad_falconera_3',
                              'tension_variador_pozo_falconera','frecuencia_pozo_falconera','factor_potencia_falconera'],
        'Funcionamiento_y_Rendimiento': ['total_horas_funcionamiento_pozo_falconera','caudal_impulsion_pozo_falconera','caudal_totalizado_pozo_falconera'],
        'Estado_del_Pozo': ['distancia_brocal_pozo_al_agua_falconera','temperatura_entrada_ull','variador_pozo_falconera']
    },
    'combinado': {
        'Temporal_y_Flujo': ['hour','weekday','week_of_year','caudal_entrada_falconera','caudal_salida_ull_playa'],
        'Calidad_y_Ubicacion': ['ph_falconera','cloro_falconera','conductividad_falconera','presion_salida_falconera','nivel_plaxiquet'],
        'Operacional_y_Clima': ['intensidad_falconera_1','potencia_pozo_falconera','factor_potencia_falconera','aemet_temperatura_media','openweather_temperatura_media']
    }
}

# =========================
# Guardar archivos por subpartición
# =========================
for cluster_name, subparts in clusters.items():
    for sub_name, cols in subparts.items():
        # Filtrar columnas existentes en df
        cols_to_save = [c for c in cols if c in df.columns]
        # Añadir columna cluster_manual si existe
        if 'cluster_manual' in df.columns:
            cols_to_save.append('cluster_manual')
        cluster_df = df[cols_to_save]
        out_file = f"../../results/execution/cluster-{cluster_name}_{sub_name}.csv"
        cluster_df.to_csv(out_file, index=False)
        print(f"Archivo guardado: {out_file}")
