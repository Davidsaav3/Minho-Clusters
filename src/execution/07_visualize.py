import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# PARÁMETROS CONFIGURABLES
INPUT_CSV = '../../results/execution/if_global.csv'   # DATASET DE ENTRADA (CON 'anomaly' Y 'is_anomaly')
RESULTS_FOLDER = '../../results/execution'           # CARPETA DE SALIDA PARA GRÁFICAS
FEATURE_TO_PLOT = 'presion_salida_falconera'                  # VARIABLE PRINCIPAL A REPRESENTAR
SHOW_ONLY_ANOMALIES = False                           # TRUE = SOLO PUNTOS ANÓMALOS
SAVE_FIGURES = True                                  # TRUE = GUARDAR FIGURAS COMO PNG
SHOW_FIGURES = False                                 # TRUE = MOSTRAR EN PANTALLA
STYLE = 'whitegrid'                                  # ESTILO SEABORN ('darkgrid', 'whitegrid', etc.)

# CREAR CARPETA DE RESULTADOS
os.makedirs(RESULTS_FOLDER, exist_ok=True)
print(f"[ INFO ] CARPETA '{RESULTS_FOLDER}' CREADA SI NO EXISTÍA")

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)
print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# ASEGURAR COLUMNAS NECESARIAS
required_cols = ['anomaly', 'is_anomaly']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"[ ERROR ] FALTA LA COLUMNA REQUERIDA '{col}' EN EL DATASET")

if FEATURE_TO_PLOT not in df.columns:
    raise ValueError(f"[ ERROR ] LA COLUMNA '{FEATURE_TO_PLOT}' NO ESTÁ EN EL DATASET")

# CONVERTIR COLUMNAS A ENTERO PARA SEABORN
df['anomaly'] = df['anomaly'].astype(int)
df['is_anomaly'] = df['is_anomaly'].astype(int)

# FILTRAR SOLO ANOMALÍAS SI ES NECESARIO
if SHOW_ONLY_ANOMALIES:
    df = df[(df['anomaly'] == 1) | (df['is_anomaly'] == 1)]
    print(f"[ INFO ] MOSTRANDO SOLO ANÓMALOS ({df.shape[0]} FILAS)")

# CONFIGURAR ESTILO DE GRÁFICAS
sns.set_style(STYLE)
plt.rcParams['figure.figsize'] = (10, 6)

# GRAFICA 1: ANOMALÍAS DETECTADAS VS REALES
plt.figure()
sns.scatterplot(
    data=df,
    x=range(len(df)),
    y=FEATURE_TO_PLOT,
    hue='anomaly',
    palette={0: 'gray', 1: 'red'},
    alpha=0.7
)
plt.title(f"ANOMALÍAS DETECTADAS VS REALES EN {FEATURE_TO_PLOT.upper()}")
plt.xlabel("ÍNDICE")
plt.ylabel(FEATURE_TO_PLOT.upper())
plt.legend(title='ANOMALÍA', labels=['NORMAL', 'DETECTADA COMO ANOMALÍA'])
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/01_anomaly_vs_real_{FEATURE_TO_PLOT}.png", dpi=300)
print("[ INFO ] Gráfico 1 generado y guardado")
if SHOW_FIGURES:
    plt.show()
plt.close()

# GRAFICA 2: DISTRIBUCIÓN DE LA VARIABLE
plt.figure()
sns.histplot(
    data=df,
    x=FEATURE_TO_PLOT,
    hue='anomaly',
    bins=50,
    kde=True,
    palette={0: 'skyblue', 1: 'red'}
)
plt.title(f"DISTRIBUCIÓN DE {FEATURE_TO_PLOT.upper()} (ANOMALÍAS VS NORMALES)")
plt.xlabel(FEATURE_TO_PLOT.upper())
plt.ylabel("FRECUENCIA")
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/02_distribution_{FEATURE_TO_PLOT}.png", dpi=300)
print("[ INFO ] Gráfico 2 generado y guardado")
if SHOW_FIGURES:
    plt.show()
plt.close()

# GRAFICA 3: BOXPLOT POR ESTADO DE ANOMALÍA
df['anomaly_str'] = df['anomaly'].astype(str)
plt.figure()
sns.boxplot(
    data=df,
    x='anomaly_str',
    y=FEATURE_TO_PLOT,
    hue='anomaly_str',          # ASIGNAR 'hue' PARA EVITAR DEPRECATION
    palette={'0': 'lightgreen', '1': 'red'},
    legend=False
)
plt.title(f"BOXPLOT DE {FEATURE_TO_PLOT.upper()} POR ESTADO DE ANOMALÍA")
plt.xlabel("DETECCIÓN IF (0=NORMAL, 1=ANOMALÍA)")
plt.ylabel(FEATURE_TO_PLOT.upper())
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/03_boxplot_{FEATURE_TO_PLOT}.png", dpi=300)
print("[ INFO ] Gráfico 3 generado y guardado")
if SHOW_FIGURES:
    plt.show()
plt.close()

# GRAFICA 4: MATRIZ DE CORRELACIÓN (TOP 10 VARIABLES)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr().abs().nlargest(10, FEATURE_TO_PLOT)[[FEATURE_TO_PLOT]]
corr = corr.sort_values(by=FEATURE_TO_PLOT, ascending=False)

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=False)
plt.title(f"TOP 10 CORRELACIONES CON {FEATURE_TO_PLOT.upper()}")
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/04_correlation_{FEATURE_TO_PLOT}.png", dpi=300)
print("[ INFO ] Gráfico 4 generado y guardado")
if SHOW_FIGURES:
    plt.show()
plt.close()

# GRAFICA 5: EVOLUCIÓN TEMPORAL (SI EXISTE COLUMNA 'datetime')
if 'datetime' in df.columns:
    plt.figure()
    sns.lineplot(
        data=df,
        x='datetime',
        y=FEATURE_TO_PLOT,
        hue='anomaly_str',
        palette={'0': 'gray', '1': 'red'}
    )
    plt.title(f"EVOLUCIÓN TEMPORAL DE {FEATURE_TO_PLOT.upper()}")
    plt.xlabel("TIEMPO")
    plt.ylabel(FEATURE_TO_PLOT.upper())
    plt.xticks(rotation=45)
    if SAVE_FIGURES:
        plt.savefig(f"{RESULTS_FOLDER}/05_temporal_{FEATURE_TO_PLOT}.png", dpi=300)
    print("[ INFO ] Gráfico 5 generado y guardado")
    if SHOW_FIGURES:
        plt.show()
    plt.close()

print("[ FINALIZADO ] GRÁFICAS GENERADAS EN CARPETA DE RESULTADOS.")
