import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

GLOBAL_CSV = '../../results/execution/04_global.csv'
CLUSTER_FILES_PATH = '../../results/execution/cluster_*.csv'
RESULTS_SUMMARY_CSV = '../../results/execution/06_results.csv'
RESULTS_FOLDER = '../../results/execution/plots'
ANOMALIES_DATASET_GLOBAL = '../../results/execution/plots/05_anomalies_dataset_global'
ANOMALIES_DATASET_SEQUENCE = '../../results/execution/plots/06_anomalies_dataset_sequence'
TYPE_HORIZONTAL_GROUPS = '../../results/execution/plots/07_type_horizontal_grups'
ANOMALIES_IF_HORIZONTAL = '../../results/execution/plots/08_anomalies_if_horizontal'
NEW_REGISTERS = '../../results/execution/plots/09_new_registers'
HORIZONTAL_VERTICAL = '../../results/execution/plots/10_horizontal_vertical'

SAVE_FIGURES = True
SHOW_FIGURES = False
STYLE = 'whitegrid'
FIGURE_SIZE = (12, 6)
LARGE_FIGURE_SIZE = (18, 6)
EXTRA_LARGE_FIGURE_SIZE = (24, 6)
HEATMAP_SIZE = (30, 18)
DPI = 300
TAB10_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

PLOT_5_ENABLED = False
PLOT_6_ENABLED = False
PLOT_7_ENABLED = False
PLOT_8_ENABLED = False
PLOT_9_ENABLED = False
PLOT_10_ENABLED = False

sns.set_style(STYLE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(ANOMALIES_DATASET_GLOBAL, exist_ok=True)
os.makedirs(ANOMALIES_DATASET_SEQUENCE, exist_ok=True)
os.makedirs(TYPE_HORIZONTAL_GROUPS, exist_ok=True)
os.makedirs(ANOMALIES_IF_HORIZONTAL, exist_ok=True)
os.makedirs(NEW_REGISTERS, exist_ok=True)
os.makedirs(HORIZONTAL_VERTICAL, exist_ok=True)

df_summary = pd.read_csv(RESULTS_SUMMARY_CSV).set_index('file')
df_if = pd.read_csv(GLOBAL_CSV)
for col in ['anomaly', 'is_anomaly', 'sequence']: df_if[col] = df_if[col].astype(int)
df_if['cluster'] = df_if.get('cluster', 0)

cluster_files = [f for f in glob.glob(CLUSTER_FILES_PATH) if not f.endswith('_if.csv')]
all_files = [GLOBAL_CSV] + cluster_files

if PLOT_5_ENABLED:
    # Columnas numéricas + columnas que empiezan por 'openweather' o 'aemet'
    numeric_columns = df_if.select_dtypes(include=[np.number]).columns.tolist()
    features_to_plot = numeric_columns 

    # Excluir columnas irrelevantes
    exclude_cols = ['anomaly', 'is_anomaly', 'sequence', 'cluster', 'datetime', 
                    'date','time','year','month','day','hour','minute','weekday',
                    'day_of_year','week_of_year','working_day','season','holiday',
                    'weekend']  # agrega más si quieres excluir otras fijas

    # Excluir también columnas que empiecen por openweather_ o aemet_
    exclude_cols += [col for col in df_if.columns if col.startswith('openweather_') or col.startswith('aemet_')]

    features_to_plot = [col for col in df_if.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

    for feature in features_to_plot:
        plt.figure(figsize=LARGE_FIGURE_SIZE)

        # Normales
        plt.scatter(df_if[(df_if['is_anomaly']==0) & (df_if['anomaly']==0)]['datetime'],
                    df_if[(df_if['is_anomaly']==0) & (df_if['anomaly']==0)][feature],
                    color='blue', alpha=0.5, s=20, label='Normal')

        # Anomalías detectadas: tamaño según sequence
        anomalies = df_if[df_if['anomaly']==1].copy()
        if not anomalies.empty:
            sizes = np.interp(anomalies['sequence'], (anomalies['sequence'].min(), anomalies['sequence'].max()), (150, 400))
            plt.scatter(anomalies['datetime'], anomalies[feature],
                        color='orange', s=sizes, alpha=0.8, label='Anomalía Detectada (tamaño = secuencia)')

        # Anomalías reales
        plt.scatter(df_if[df_if['is_anomaly']==1]['datetime'],
                    df_if[df_if['is_anomaly']==1][feature],
                    color='red', marker='X', s=60, alpha=0.7, label='Anomalía Real')

        plt.title(f"Anomalías Reales vs Detectadas: {feature.upper()}")
        plt.xlabel("Datetime")
        plt.ylabel(feature.upper())
        plt.xticks(rotation=45)
        plt.legend(title='Leyenda', loc='upper right')
        plt.annotate("Tamaño de punto = longitud de la secuencia", xy=(0.01, -0.15),
                     xycoords='axes fraction', fontsize=10, color='orange', ha='left')

        if SAVE_FIGURES:
            plt.savefig(f"{ANOMALIES_DATASET_GLOBAL}/{feature}.png", dpi=DPI, bbox_inches='tight')
        if SHOW_FIGURES:
            plt.show()
        plt.close()
        print(f" [ GRÁFICO 1 ] {feature}.png")
