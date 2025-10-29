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
SUBFOLDER_ANOMALIES = '../../results/execution/plots/01_anomalias'
SUBFOLDER_FREQUENCY = '../../results/execution/plots/02_frequencia'
SUBFOLDER_SEQUENCES = '../../results/execution/plots/03_secuencias'
SUBFOLDER_ANOMALY_PLOT = '../../results/execution/plots/04_score'
SUBFOLDER_ANOMALY_PROPORTION = '../../results/execution/plots/'

SAVE_FIGURES = True
SHOW_FIGURES = False
STYLE = 'whitegrid'
FIGURE_SIZE = (12, 6)
LARGE_FIGURE_SIZE = (18, 6)
EXTRA_LARGE_FIGURE_SIZE = (24, 6)
HEATMAP_SIZE = (30, 18)
DPI = 300
TAB10_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

PLOT_1_ENABLED = True
PLOT_2_ENABLED = False
PLOT_3_ENABLED = False
PLOT_4_ENABLED = False
PLOT_5_ENABLED = False
PLOT_6_ENABLED = False
PLOT_7_ENABLED = False
PLOT_8_ENABLED = False
PLOT_9_ENABLED = False
PLOT_10_ENABLED = False
PLOT_11_ENABLED = False
PLOT_12_ENABLED = False
PLOT_13_ENABLED = False
PLOT_14_ENABLED = False

sns.set_style(STYLE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(SUBFOLDER_ANOMALIES, exist_ok=True)
os.makedirs(SUBFOLDER_FREQUENCY, exist_ok=True)
os.makedirs(SUBFOLDER_SEQUENCES, exist_ok=True)
os.makedirs(SUBFOLDER_ANOMALY_PLOT, exist_ok=True)
os.makedirs(SUBFOLDER_ANOMALY_PROPORTION, exist_ok=True)

df_summary = pd.read_csv(RESULTS_SUMMARY_CSV).set_index('file')
df_if = pd.read_csv(GLOBAL_CSV)
for col in ['anomaly', 'is_anomaly', 'sequence']: df_if[col] = df_if[col].astype(int)
df_if['cluster'] = df_if.get('cluster', 0)

cluster_files = [f for f in glob.glob(CLUSTER_FILES_PATH) if not f.endswith('_if.csv')]
all_files = [GLOBAL_CSV] + cluster_files

if PLOT_1_ENABLED:
    numeric_columns = df_if.select_dtypes(include=[np.number]).columns.tolist()
    features_to_plot = numeric_columns 

    exclude_cols = ['anomaly', 'is_anomaly', 'sequence', 'cluster', 'datetime', 
                    'date','time','year','month','day','hour','minute','weekday',
                    'day_of_year','week_of_year','working_day','season','holiday',
                    'weekend']
    exclude_cols += [col for col in df_if.columns if col.startswith('openweather_') or col.startswith('aemet_')]

    features_to_plot = [col for col in df_if.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

    # Recuento de intersecciones
    detectadas = df_if['anomaly']==1
    genuinas = df_if['genuine_anomaly']==1
    reales = df_if['is_anomaly']==1

    count_detectadas = detectadas.sum()
    count_genuinas = genuinas.sum()
    count_reales = reales.sum()

    count_detect_genu = (detectadas & genuinas).sum()
    count_detect_real = (detectadas & reales).sum()
    count_genu_real = (genuinas & reales).sum()
    count_all_three = (detectadas & genuinas & reales).sum()

    for feature in features_to_plot:
        plt.figure(figsize=LARGE_FIGURE_SIZE)

        # Normales
        plt.scatter(df_if[~detectadas & ~reales]['datetime'],
                    df_if[~detectadas & ~reales][feature],
                    color='blue', alpha=0.5, s=20, label='Normal')

        # Detectadas
        anomalies = df_if[detectadas].copy()
        if not anomalies.empty:
            sizes = np.interp(anomalies['sequence'] if 'sequence' in df_if.columns else np.ones(len(anomalies)),
                              (anomalies['sequence'].min() if 'sequence' in df_if.columns else 1, 
                               anomalies['sequence'].max() if 'sequence' in df_if.columns else 1), 
                              (150, 400))
            plt.scatter(anomalies['datetime'], anomalies[feature],
                        color='orange', s=sizes, alpha=0.5,
                        label=f'Detectadas ({count_detectadas})')

        # Genuinas
        genuine_anomalies = df_if[genuinas].copy()
        if not genuine_anomalies.empty:
            plt.scatter(genuine_anomalies['datetime'], genuine_anomalies[feature],
                        color='green', marker='o', s=80, alpha=0.5,
                        label=f'Genuinas ({count_genuinas})')

        # Reales
        reales_df = df_if[reales].copy()
        if not reales_df.empty:
            plt.scatter(reales_df['datetime'], reales_df[feature],
                        color='red', marker='X', s=60, alpha=0.5,
                        label=f'Reales ({count_reales})')

        # Agregar anotaciones de intersección en la leyenda
        inter_label = (f"Intersecciones: Detect&Genu={count_detect_genu}, "
                       f"Detect&Real={count_detect_real}, Genu&Real={count_genu_real}, "
                       f"Todas= {count_all_three}")
        plt.title(f"Anomalías Reales vs Detectadas: {feature.upper()}")
        plt.xlabel("Datetime")
        plt.ylabel(feature.upper())
        plt.xticks(rotation=45)
        plt.legend(title='Leyenda', loc='upper right')
        plt.annotate(inter_label, xy=(0.01, -0.15), xycoords='axes fraction', fontsize=10, color='black', ha='left')
        plt.annotate("Tamaño de punto = longitud de la secuencia", xy=(0.01, -0.20), xycoords='axes fraction', fontsize=10, color='orange', ha='left')

        if SAVE_FIGURES:
            plt.savefig(f"{SUBFOLDER_ANOMALIES}/{feature}.png", dpi=DPI, bbox_inches='tight')
        if SHOW_FIGURES:
            plt.show()
        plt.close()
        print(f" [ GRÁFICO 1 ] {feature}.png")



if PLOT_2_ENABLED:
    for file in cluster_files:
        df_cluster = pd.read_csv(file)
        for col in ['anomaly', 'is_anomaly', 'sequence']: df_cluster[col] = df_cluster[col].astype(int)
        cluster_name = os.path.splitext(os.path.basename(file))[0]
        plt.figure(figsize=FIGURE_SIZE)
        sns.histplot(df_if['anomaly_score'], bins=50, color='blue', alpha=0.4, label='Global', kde=False)
        sns.histplot(df_cluster['anomaly_score'], bins=50, color='red', alpha=0.6, label=cluster_name, kde=False)
        plt.title(f"Distribución del Score de Anomalía - {cluster_name}")
        plt.xlabel("Score de Anomalía")
        plt.ylabel("Frecuencia")
        plt.legend()
        if SAVE_FIGURES: plt.savefig(os.path.join(SUBFOLDER_FREQUENCY, f"{cluster_name}_frecuencia.png"), dpi=DPI)
        plt.close()
        print(f" [ GRÁFICO 2 ] {cluster_name}_frecuencia.png")

if PLOT_3_ENABLED:
    for file in all_files:
        df_cluster = pd.read_csv(file)
        cluster_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
        if len(df_cluster) == len(df_if):
            df_cluster['cluster'] = df_if['cluster'].values
        else:
            df_cluster['cluster'] = cluster_name

        df_cluster['sequence'] = df_cluster.get('sequence', 0)
        df_sequences = df_cluster[df_cluster['sequence'] > 0].copy() if df_cluster['sequence'].sum() > 0 else df_cluster.copy()

        # DataFrame para el gráfico
        df_plot = df_sequences.groupby(['datetime', 'cluster'], as_index=False)['sequence'].sum().sort_values('datetime')
        if df_plot.empty: 
            df_plot = pd.DataFrame({'datetime': ['no_data'], 'cluster': ['none'], 'sequence': [0]})

        plt.figure(figsize=EXTRA_LARGE_FIGURE_SIZE)
        # Barras por cluster
        sns.barplot(data=df_plot, x='datetime', y='sequence', hue='cluster', palette='tab10', dodge=True, edgecolor=None, width=2)

        # Marcar anomalías reales
        anomalies = df_sequences[df_sequences['is_anomaly'] == 1]
        if not anomalies.empty:
            # Ubicamos los puntos sobre la barra
            for idx, row in anomalies.iterrows():
                x_pos = list(df_plot['datetime']).index(row['datetime'])  # índice x de la barra
                plt.scatter(x=x_pos, y=row['sequence'], color='red', s=60, zorder=10, label='Anomalía real')

        plt.title(f"Distribución de Secuencias de Anomalía por Cluster - {cluster_name}")
        plt.xlabel("Datetime")
        plt.ylabel("Tamaño Total de Secuencias")

        # Limitar número de labels
        max_labels = 20
        step = max(1, len(df_plot['datetime'].unique()) // max_labels)
        plt.xticks(ticks=range(0, len(df_plot['datetime'].unique()), step), labels=df_plot['datetime'].unique()[::step], rotation=45)

        # Leyenda
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # evita duplicados
        plt.legend(by_label.values(), by_label.keys(), title='Cluster / Anomalía', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        if SAVE_FIGURES: 
            plt.savefig(os.path.join(SUBFOLDER_SEQUENCES, f"{cluster_name}_secuencias.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 3 ] {cluster_name}_secuencias.png")


if PLOT_4_ENABLED:
    for file in all_files:
        df_cluster = pd.read_csv(file)
        cluster_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
        if len(df_cluster) == len(df_if): df_cluster['cluster'] = df_if['cluster'].values
        else: df_cluster['cluster'] = cluster_name
        df_cluster['anomaly'] = df_cluster.get('anomaly', 0).astype(int)
        plt.figure(figsize=LARGE_FIGURE_SIZE)
        sns.scatterplot(data=df_cluster, x='datetime', y='anomaly_score', hue='cluster', palette='tab10', size='sequence', sizes=(20, 200), alpha=0.6)
        anomalies = df_cluster[df_cluster['anomaly'] == 1]
        if not anomalies.empty:
            sizes = np.interp(anomalies['sequence'], (df_cluster['sequence'].min(), df_cluster['sequence'].max()), (10, 100))
            plt.scatter(anomalies['datetime'], anomalies['anomaly_score'], c='red', s=sizes, marker='X', edgecolor='black', linewidth=0.3, label='Real Anomaly')
        plt.title(f"Distribución temporal de scores - {cluster_name}")
        plt.xlabel("Datetime")
        plt.ylabel("Score de Anomalía")
        plt.xticks(rotation=45)
        plt.legend(title='Cluster / Anomalía', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if SAVE_FIGURES: plt.savefig(os.path.join(SUBFOLDER_ANOMALY_PLOT, f"{cluster_name}_anomalia.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 4 ] {cluster_name}_anomalia.png")

if PLOT_5_ENABLED:
    try:
        df_global = pd.read_csv(GLOBAL_CSV)
        df_global['cluster'] = df_global['cluster'].astype(str)
        df_resultados = pd.DataFrame()
        files_to_process = [GLOBAL_CSV] + cluster_files
        for file in files_to_process:
            subset_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
            df_subset = pd.read_csv(file)
            df_subset['anomaly'] = df_subset['anomaly'].astype(int)
            if len(df_subset) != len(df_global): 
                print(f"[AVISO] {subset_name} tiene diferente número de filas que el global, se omite.")
                continue
            df_subset['cluster'] = df_global['cluster']
            prop_pct = df_subset.groupby('cluster')['anomaly'].mean().multiply(100).rename(subset_name)
            df_resultados = pd.concat([df_resultados, prop_pct], axis=1)
        if df_resultados.empty: print(" [ GRÁFICO 5 ] No se pudieron calcular proporciones de anomalías.")
        else:
            df_plot = df_resultados.T.fillna(0)
            plt.figure(figsize=(20, 8))
            df_plot.plot(kind='bar', cmap='tab10')
            plt.title("Porcentaje de Anomalías por Cluster Global en cada archivo (incluye 'global')")
            plt.xlabel("Archivo / Subconjunto")
            plt.ylabel("Proporción de Anomalías (%)")
            plt.legend(title='Cluster Global', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            if SAVE_FIGURES: plt.savefig(os.path.join(SUBFOLDER_ANOMALY_PROPORTION, "05_proporcion_anomalias_por_cluster.png"), dpi=DPI, bbox_inches='tight')
            plt.close()
            print(" [ GRÁFICO 5 ] proporcion_anomalias_por_cluster.png")
    except Exception as e: print(f" [ GRÁFICO 5 ] Error: {e}")

if any([PLOT_11_ENABLED, PLOT_12_ENABLED, PLOT_13_ENABLED]):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 18))
    if PLOT_11_ENABLED:
        metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'mcc']
        df_summary[metrics].plot(kind='bar', ax=axes[0])
        axes[0].set_title("Métricas de Rendimiento")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0, 1.1)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(title='Metric')
    if PLOT_12_ENABLED:
        ratio_metrics = ['anomalies_real', 'anomalies_detected', 'detections_correct', 'total_coincidences']
        df_summary[ratio_metrics].plot(kind='bar', ax=axes[1])
        axes[1].set_title("Ratio de Anomalías Reales y Detectadas")
        axes[1].set_ylabel("Ratio")
        axes[1].tick_params(axis='x', rotation=45)
    if PLOT_13_ENABLED:
        df_summary[['detections_correct', 'false_positives', 'false_negatives']].plot(kind='bar', ax=axes[2], color=['blue', 'green', 'red'])
        axes[2].set_title("Verdaderas positivas, falsos positivos y falsos negativos")
        axes[2].set_ylabel("Número")
        axes[2].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/11_12_13_combinado.png", dpi=DPI)
    plt.close()
    print(f" [ GRÁFICO ] 11_12_13_combinado.png")

if PLOT_14_ENABLED:
    numeric_cols = df_if.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=HEATMAP_SIZE)
    sns.heatmap(df_if[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True, annot_kws={"size": 3}, linewidths=0.3, linecolor='white')
    plt.title("Correlation Matrix - All Numeric Features", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=4)
    plt.yticks(rotation=0, fontsize=4)
    if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/14_correlation_matrix.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f" [ GRÁFICO 14 ] 14_correlation_matrix.png")