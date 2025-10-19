import pandas as pd                  # IMPORTA PANDAS PARA MANEJO DE DATOS
import matplotlib.pyplot as plt      # IMPORTA MATPLOTLIB PARA CREAR GRÁFICAS
import seaborn as sns                # IMPORTA SEABORN PARA ESTILO Y VISUALIZACIÓN AVANZADA
import os                            # IMPORTA OS PARA GESTIONAR RUTAS Y CARPETAS
import glob                          # IMPORTA GLOB PARA BUSCAR ARCHIVOS CON PATRONES

# DEFINE LAS RUTAS Y PARÁMETROS GENERALES DEL SCRIPT
IF_GLOBAL_CSV = '../../results/execution/if_global.csv'        # ARCHIVO GLOBAL CON RESULTADOS DEL ISOLATION FOREST
IF_01_CSV = '../../results/execution/01_if.csv'                # ARCHIVO CON SECUENCIAS DE ANOMALÍAS DETECTADAS
CLUSTER_FILES_PATH = '../../results/execution/cluster_*.csv'   # PATRÓN DE BÚSQUEDA DE ARCHIVOS POR CLUSTER
RESULTS_SUMMARY_CSV = '../../results/execution/06_results.csv' # ARCHIVO CON LAS MÉTRICAS DE RENDIMIENTO
RESULTS_FOLDER = '../../results/execution/plots'               # CARPETA DONDE SE GUARDARÁN LOS GRÁFICOS

FEATURE_TO_PLOT = 'nivel_plaxiquet'                           # VARIABLE PRINCIPAL A VISUALIZAR EN LOS GRÁFICOS
SAVE_FIGURES = True                                            # DEFINE SI SE GUARDAN LAS FIGURAS EN ARCHIVOS
SHOW_FIGURES = False                                           # DEFINE SI SE MUESTRAN EN PANTALLA LAS FIGURAS
STYLE = 'whitegrid'                                            # ESTILO VISUAL PARA SEABORN

# CREA LA CARPETA DE RESULTADOS SI NO EXISTE
os.makedirs(RESULTS_FOLDER, exist_ok=True)                     # CREA LA CARPETA SI NO EXISTE
print(f"[ INFO ] CARPETA '{RESULTS_FOLDER}' CREADA SI NO EXISTÍA")  # IMPRIME CONFIRMACIÓN EN CONSOLA

# CONFIGURA EL ESTILO Y EL TAMAÑO DE LAS FIGURAS
sns.set_style(STYLE)                                           # APLICA ESTILO VISUAL A SEABORN
plt.rcParams['figure.figsize'] = (12, 6)                       # DEFINE TAMAÑO POR DEFECTO DE LAS FIGURAS

# CARGA EL ARCHIVO PRINCIPAL CON LOS RESULTADOS DEL MODELO GLOBAL
df_if = pd.read_csv(IF_GLOBAL_CSV)                             # LEE EL ARCHIVO CSV PRINCIPAL

# CONVIERTE VARIABLES A TIPO ENTERO PARA ASEGURAR CONSISTENCIA
df_if['anomaly'] = df_if['anomaly'].astype(int)                # CONVIERTE LA COLUMNA 'anomaly' A ENTERO
df_if['is_anomaly'] = df_if['is_anomaly'].astype(int)          # CONVIERTE LA COLUMNA 'is_anomaly' A ENTERO
df_if['sequence'] = df_if['sequence'].astype(int)              # CONVIERTE LA COLUMNA 'sequence' A ENTERO

# AÑADE LA COLUMNA CLUSTER SI NO EXISTE EN EL DATAFRAME
if 'cluster' not in df_if.columns:                             # VERIFICA SI FALTA LA COLUMNA
    df_if['cluster'] = 0                                       # CREA COLUMNA 'cluster' CON VALOR 0

# ORDENA EL DATAFRAME POR CLUSTER Y FECHA
df_if_sorted = df_if.sort_values(['cluster', 'datetime'])      # ORDENA LOS REGISTROS PARA ANÁLISIS TEMPORAL

# 1. GRÁFICO: ANOMALÍAS DETECTADAS VS REALES
plt.figure(figsize=(18, 6))                                    # CREA UNA NUEVA FIGURA DE TAMAÑO GRANDE
sns.scatterplot(                                               
    data=df_if,                                                 # USA EL DATAFRAME GLOBAL
    x='datetime',                                               # EJE X: FECHA Y HORA
    y=FEATURE_TO_PLOT,                                          # EJE Y: VARIABLE PRINCIPAL
    hue='anomaly',                                              # COLOR SEGÚN SI ES ANOMALÍA O NO
    palette={0: 'gray', 1: 'red'},                              # DEFINE COLORES (NORMAL=GRIS, ANOMALÍA=ROJO)
    alpha=0.7                                                   # DEFINE TRANSPARENCIA
)
plt.title(f"Anomalies Detected vs Real: {FEATURE_TO_PLOT.upper()}")  # TÍTULO DEL GRÁFICO
plt.xlabel("Datetime")                                          # ETIQUETA DEL EJE X
plt.ylabel(FEATURE_TO_PLOT.upper())                             # ETIQUETA DEL EJE Y
plt.xticks(rotation=45)                                         # ROTA ETIQUETAS DE TIEMPO
plt.legend(title='Anomaly', labels=['Normal', 'Detected'])      # DEFINE LEYENDA
if SAVE_FIGURES:                                                # SI SE DEBE GUARDAR LA FIGURA
    plt.savefig(f"{RESULTS_FOLDER}/01_anomalies_vs_real.png", dpi=300, bbox_inches='tight')  # GUARDA EL GRÁFICO
plt.close()                                                     # CIERRA LA FIGURA PARA LIBERAR MEMORIA

# 2. HEATMAP DE CORRELACIÓN ENTRE VARIABLES NUMÉRICAS
numeric_cols = df_if.select_dtypes(include=['float64', 'int64']).columns  # SELECCIONA SOLO COLUMNAS NUMÉRICAS
plt.figure(figsize=(30, 18))                                  # CREA UNA FIGURA DE GRAN TAMAÑO
sns.heatmap(                                                  # CREA UN HEATMAP DE CORRELACIONES
    df_if[numeric_cols].corr(),                               # CALCULA MATRIZ DE CORRELACIÓN
    annot=True, fmt=".2f", cmap='coolwarm',                   # MUESTRA VALORES NUMÉRICOS CON FORMATO
    cbar=True, annot_kws={"size": 3}, linewidths=0.3, linecolor='white'  # CONFIGURA DETALLES VISUALES
)
plt.title("Correlation Matrix - All Numeric Features", fontsize=14)  # TÍTULO DEL HEATMAP
plt.xticks(rotation=45, ha='right', fontsize=4)               # ROTA Y AJUSTA ETIQUETAS X
plt.yticks(rotation=0, fontsize=4)                            # AJUSTA ETIQUETAS Y
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/02_correlation_matrix.png", dpi=300, bbox_inches='tight')  # GUARDA EL HEATMAP
plt.close()

# 3. HISTOGRAMA DE SCORES DE ANOMALÍAS
plt.figure()                                                 # CREA NUEVA FIGURA
sns.histplot(df_if['anomaly_score'], bins=50, kde=True, color='blue')  # CREA HISTOGRAMA CON CURVA KDE
plt.title("Distribution of Anomaly Scores")                  # TÍTULO DEL GRÁFICO
plt.xlabel("Anomaly Score")                                  # ETIQUETA EJE X
plt.ylabel("Frequency")                                      # ETIQUETA EJE Y
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/03_anomaly_score_distribution.png", dpi=300)  # GUARDA EL HISTOGRAMA
plt.close()

# 4. SECUENCIAS DE ANOMALÍAS EN EL TIEMPO
df_if_01 = pd.read_csv(IF_01_CSV)                             # CARGA EL ARCHIVO DE ANÁLISIS DE SECUENCIAS
if 'cluster' not in df_if_01.columns:                         # SI NO TIENE COLUMNA CLUSTER
    df_if_01['cluster'] = 0                                   # AÑADELA CON VALOR 0
df_sequences = df_if_01[df_if_01['sequence'] > 0]             # FILTRA SOLO REGISTROS CON SECUENCIAS ACTIVAS

plt.figure(figsize=(18, 6))                                   # CREA NUEVA FIGURA
sns.scatterplot(                                              # DIBUJA PUNTOS TEMPORALES DE SECUENCIAS
    data=df_sequences, x='datetime', y='sequence', hue='cluster',
    palette='tab10', size='sequence', sizes=(20, 200), alpha=0.7
)
plt.title("Temporal Distribution of Repeated Anomalies (Sequence Length)")  # TÍTULO
plt.xlabel("Datetime")                                        # ETIQUETA X
plt.ylabel("Anomaly Sequence Length")                         # ETIQUETA Y
plt.xticks(rotation=45)                                       # ROTA FECHAS
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')  # POSICIONA LEYENDA
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/04_anomaly_sequence_over_time.png", dpi=300, bbox_inches='tight')  # GUARDA FIGURA
plt.close()

# CARGA Y COMBINA LOS ARCHIVOS DE CLUSTERS
cluster_files = glob.glob(CLUSTER_FILES_PATH)                 # OBTIENE TODOS LOS ARCHIVOS QUE CUMPLEN EL PATRÓN
dfs_clusters = []                                             # LISTA PARA ALMACENAR LOS DATAFRAMES

for file in cluster_files:                                    # RECORRE CADA ARCHIVO ENCONTRADO
    if not file.endswith('_if.csv'):                          # IGNORA LOS ARCHIVOS DE RESULTADOS IF
        dfc = pd.read_csv(file)                               # LEE EL ARCHIVO CSV
        if 'datetime' not in dfc.columns and 'timestamp' in dfc.columns:
            dfc['datetime'] = dfc['timestamp']                # CREA COLUMNA DATETIME SI NO EXISTE
        if 'cluster' not in dfc.columns:
            dfc['cluster'] = 0                                # AÑADE COLUMNA CLUSTER SI FALTA
        dfs_clusters.append(dfc)                              # AÑADE EL DATAFRAME A LA LISTA

df_all_clusters = pd.concat(dfs_clusters, ignore_index=True)  # UNE TODOS LOS DATAFRAMES EN UNO SOLO

# 5. SCATTER DETALLADO CON LEYENDA 
plt.figure(figsize=(12, 6))  # Crea una figura con tamaño moderado para visualizar bien la dispersión
sns.scatterplot(
    data=df_all_clusters,
    x='sequence',                        # Eje X: longitud de las secuencias detectadas
    y='anomaly_score',                   # Eje Y: puntuación de anomalía (score)
    hue='cluster',                       # Color por cluster para distinguir agrupaciones
    palette='tab10',                     # Paleta de colores para los clusters
    alpha=0.6,                           # Transparencia de los puntos
    size='sequence',                     # Tamaño de los puntos proporcional a la longitud de la secuencia
    sizes=(20, 200)                      # Rango de tamaños posibles para los puntos
)
plt.title("Anomaly Score vs Sequence Length by Cluster")  # Título descriptivo del gráfico
plt.xlabel("Sequence Length")                             # Etiqueta eje X
plt.ylabel("Anomaly Score")                               # Etiqueta eje Y
plt.legend(title='Cluster', loc='upper right')  # <- aquí
if SAVE_FIGURES:                                           # Si está activada la opción de guardar
    plt.savefig(f"{RESULTS_FOLDER}/05_sequence_vs_score_by_cluster.png", dpi=300, bbox_inches='tight')  # Guarda la figura en alta resolución
plt.close()                                                # Cierra la figura para liberar memoria

# 6. DISTRIBUCIÓN TEMPORAL DE SCORES POR CLUSTER
plt.figure(figsize=(18, 6))  # Figura panorámica para visualizar el eje temporal
sns.scatterplot(
    data=df_all_clusters,
    x='datetime',                         # Eje X: fecha y hora de cada observación
    y='anomaly_score',                    # Eje Y: puntuación de anomalía
    hue='cluster',                        # Color según el cluster asignado
    palette='tab10',                      # Paleta de colores estándar
    size='sequence',                      # Tamaño de puntos según longitud de secuencia
    sizes=(20, 200),
    alpha=0.6                             # Transparencia para evitar saturación
)
plt.title("Temporal Distribution of Anomaly Scores per Cluster")  # Título del gráfico
plt.xlabel("Datetime")                                            # Etiqueta eje X
plt.ylabel("Anomaly Score")                                       # Etiqueta eje Y
plt.xticks(rotation=45)                                           # Rota etiquetas del eje X para mejor lectura
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')  # Coloca la leyenda fuera del gráfico
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/06_temporal_score_by_cluster.png", dpi=300, bbox_inches='tight')  # Guarda la figura
plt.close()

# CARGA MÉTRICAS DE RENDIMIENTO
df_summary = pd.read_csv(RESULTS_SUMMARY_CSV)          # Carga el CSV con las métricas calculadas de los modelos
df_summary.set_index('file', inplace=True)             # Usa el nombre del archivo (método o modelo) como índice

# 7. GRÁFICO DE MÉTRICAS DE RENDIMIENTO
metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'mcc']  # Lista de métricas a visualizar
df_summary[metrics].plot(kind='bar', figsize=(16, 6))              # Gráfico de barras comparando el rendimiento por método
plt.title("Performance Metrics per Method / File")                 # Título del gráfico
plt.ylabel("Score")                                                # Etiqueta eje Y
plt.ylim(0, 1.1)                                                   # Escala del eje Y de 0 a 1.1 para ver bien las diferencias
plt.xticks(rotation=45, ha='right')                                # Rota las etiquetas X
plt.legend(title='Metric')                                         # Muestra la leyenda de métricas
plt.tight_layout()                                                 # Ajusta márgenes
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/07_metrics_comparison.png", dpi=300)  # Guarda la figura
plt.close()

# 8. RATIO DE DETECCIÓN VS FALSOS POSITIVOS
ratio_metrics = ['ratio_detection', 'ratio_fp']                    # Selecciona las métricas de ratio
df_summary[ratio_metrics].plot(
    kind='bar',
    figsize=(16, 6),
    color=['green', 'red']                                         # Verde para detección, rojo para falsos positivos
)
plt.title("Ratio Detection vs False Positives")                    # Título del gráfico
plt.ylabel("Ratio")                                                # Etiqueta eje Y
plt.xticks(rotation=45, ha='right')                                # Rota etiquetas X
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/08_ratio_detection_fp.png", dpi=300)  # Guarda la figura
plt.close()

# 9. ANOMALÍAS DETECTADAS VS COINCIDENCIAS TOTALES
df_summary[['anomalies_detected', 'total_coincidences']].plot(
    kind='bar',
    figsize=(16, 6),
    color=['blue', 'gray']                                         # Azul para anomalías detectadas, gris para coincidencias
)
plt.title("Anomalies Detected vs Total Coincidences")              # Título del gráfico
plt.ylabel("Count")                                                # Etiqueta eje Y
plt.xticks(rotation=45, ha='right')                                # Rota etiquetas X
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/09_detected_vs_coincidences.png", dpi=300)  # Guarda la figura
plt.close()

# MENSAJE FINAL DE CONFIRMACIÓN
print("Visualizations saved in:", RESULTS_FOLDER)  # Mensaje en consola confirmando la ubicación de las gráficas
