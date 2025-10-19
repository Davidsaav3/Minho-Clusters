import pandas as pd                  # IMPORTA PANDAS PARA MANEJO DE DATOS
import matplotlib.pyplot as plt      # IMPORTA MATPLOTLIB PARA CREAR GRÁFICAS
import seaborn as sns                # IMPORTA SEABORN PARA ESTILO Y VISUALIZACIÓN AVANZADA
import os                            # IMPORTA OS PARA GESTIONAR RUTAS Y CARPETAS
import glob                          # IMPORTA GLOB PARA BUSCAR ARCHIVOS CON PATRONES
import numpy as np

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

# 1. GRÁFICO: ANOMALÍAS DETECTADAS VS REALES
plt.figure(figsize=(18, 6))                                    # CREA UNA NUEVA FIGURA DE TAMAÑO GRANDE
sns.scatterplot(                                               
    data=df_if,                                                 # USA EL DATAFRAME GLOBAL
    x='datetime',                                               # EJE X: FECHA Y HORA
    y=FEATURE_TO_PLOT,                                          # EJE Y: VARIABLE PRINCIPAL
    hue='anomaly',                                              # COLOR SEGÚN SI ES ANOMALÍA O NO
    palette={0: 'blue', 1: 'red'},                              # DEFINE COLORES (NORMAL=GRIS, ANOMALÍA=ROJO)
    alpha=0.7                                                   # DEFINE TRANSPARENCIA
)
plt.title(f"Anomalies Detected vs Real: {FEATURE_TO_PLOT.upper()}")  # TÍTULO DEL GRÁFICO
plt.xlabel("Datetime")                                          # ETIQUETA DEL EJE X
plt.ylabel(FEATURE_TO_PLOT.upper())                             # ETIQUETA DEL EJE Y
plt.xticks(rotation=45)                                         # ROTA ETIQUETAS DE TIEMPO
plt.legend(title='Anomaly', labels=['Real', 'Detected'])      # DEFINE LEYENDA
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
df_if = pd.read_csv(IF_01_CSV)                             # LEE EL ARCHIVO CSV PRINCIPAL

plt.figure()                                                 # CREA NUEVA FIGURA
sns.histplot(df_if['anomaly_score'], bins=50, kde=True, color='red')  # CREA HISTOGRAMA CON CURVA KDE
plt.title("Distribution of Anomaly Scores")                  # TÍTULO DEL GRÁFICO
plt.xlabel("Anomaly Score")                                  # ETIQUETA EJE X
plt.ylabel("Frequency")                                      # ETIQUETA EJE Y
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/03_anomaly_score_distribution.png", dpi=300)  # GUARDA EL HISTOGRAMA
plt.close()



# Rutas y parámetros
CLUSTER_FILES_PATH = '../../results/execution/cluster_*.csv'
RESULTS_FOLDER = '../../results/execution/plots'
SAVE_FIGURES = True
target_cluster_name = "cluster_ubicacion_Ull_Pueblo.csv"  # Cluster específico

# Buscar archivo del cluster
cluster_files = glob.glob(CLUSTER_FILES_PATH)
selected_files = [f for f in cluster_files if target_cluster_name in os.path.basename(f)]

if selected_files:
    cluster_file = selected_files[0]
    df_cluster = pd.read_csv(cluster_file)

    # Asegurar columna 'sequence'
    if 'sequence' not in df_cluster.columns:
        df_cluster['sequence'] = 0

    # Filtrar solo secuencias activas
    df_sequences = df_cluster[df_cluster['sequence'] > 0].copy()

    # Agrupar por valor textual de 'datetime'
    df_plot = df_sequences.groupby('datetime')['sequence'].sum().reset_index()

    # Crear gráfico de barras doble ancho
    plt.figure(figsize=(24, 6))
    sns.barplot(
        data=df_plot,
        x='datetime',
        y='sequence',
        color='skyblue'
    )
    plt.title(f"Temporal Distribution of Repeated Anomalies (Sequence Length) - {target_cluster_name}")
    plt.xlabel("Datetime")
    plt.ylabel("Total Anomaly Sequence Length")
    plt.xticks(rotation=90)  # Rota etiquetas X para mejor lectura

    # Guardar figura
    if SAVE_FIGURES:
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        plt.savefig(f"{RESULTS_FOLDER}/04_anomaly_sequence_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ INFO ] Gráfico 4 guardado para el cluster: {target_cluster_name}")
else:
    print(f"[ WARN ] No se encontraron archivos de clusters con '{target_cluster_name}' en el nombre.")



dfs_clusters = []    
# 5 y 6 CARGA Y COMBINA LOS ARCHIVOS DE CLUSTERS
for file in selected_files:                                    # RECORRE CADA ARCHIVO ENCONTRADO
    if not file.endswith('_if.csv'):                          # IGNORA LOS ARCHIVOS DE RESULTADOS IF
        dfc = pd.read_csv(file)                               # LEE EL ARCHIVO CSV
        if 'datetime' not in dfc.columns and 'timestamp' in dfc.columns:
            dfc['datetime'] = dfc['timestamp']                # CREA COLUMNA DATETIME SI NO EXISTE
        if 'cluster' not in dfc.columns:
            cluster_name = "cluster_" + os.path.splitext(os.path.basename(file))[0]
            dfc['cluster'] = cluster_name                     # AÑADE COLUMNA CLUSTER CON NOMBRE DINÁMICO
        dfs_clusters.append(dfc)                              # AÑADE EL DATAFRAME A LA LISTA

# CONCATENA TODOS LOS DATAFRAMES EN UNO SOLO
df_all_clusters = pd.concat(dfs_clusters, ignore_index=True)

# 5. SCATTER DETALLADO CON LEYENDA
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df_all_clusters,
    x='sequence',                        
    y='anomaly_score',                   
    hue='cluster',                       
    palette='tab10',                     
    alpha=0.6,                           
    size='sequence',                     
    sizes=(20, 200)                      
)
plt.title("Anomaly Score vs Sequence Length by Cluster")
plt.xlabel("Sequence Length")
plt.ylabel("Anomaly Score")
plt.legend(title='Cluster', loc='upper right')
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/05_sequence_vs_score_by_cluster.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. DISTRIBUCIÓN TEMPORAL DE SCORES POR CLUSTER
plt.figure(figsize=(18, 6))
sns.scatterplot(
    data=df_all_clusters,
    x='datetime',                         
    y='anomaly_score',                    
    hue='cluster',                        
    palette='tab10',                      
    size='sequence',                     
    sizes=(20, 200),
    alpha=0.6                             
)
plt.title("Temporal Distribution of Anomaly Scores per Cluster")
plt.xlabel("Datetime")
plt.ylabel("Anomaly Score")
plt.xticks(rotation=45)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/06_temporal_score_by_cluster.png", dpi=300, bbox_inches='tight')
plt.close()