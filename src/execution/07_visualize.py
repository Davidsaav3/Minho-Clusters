import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

# Define paths and parameters
GLOBAL_CSV = '../../results/execution/04_global.csv'
CLUSTER_FILES_PATH = '../../results/execution/cluster_*.csv'
RESULTS_SUMMARY_CSV = '../../results/execution/06_results.csv'
RESULTS_FOLDER = '../../results/execution/plots'
SUBFOLDER_ANOMALIES = '../../results/execution/plots/01_anomalias'
SUBFOLDER_FREQUENCY = '../../results/execution/plots/02_frequencia_anomalias'
SUBFOLDER_SEQUENCES = '../../results/execution/plots/03_distribucion_secuencias'
SUBFOLDER_SEQUENCE_PLOT = '../../results/execution/plots/04_sequencia'
SUBFOLDER_ANOMALY_PLOT = '../../results/execution/plots/05_anomalia'
SUBFOLDER_ANOMALY_TIMESERIES = '../../results/execution/plots/06_anomaly_timeseries'
SUBFOLDER_ANOMALY_DENSITY = '../../results/execution/plots/07_anomaly_density'
SUBFOLDER_ANOMALY_PROPORTION = '../../results/execution/plots/08_anomaly_proportion'
SUBFOLDER_SEQUENCE_VIOLIN = '../../results/execution/plots/09_sequence_violin'
SUBFOLDER_CLUSTER_PAIRPLOT = '../../results/execution/plots/10_cluster_pairplot'

SAVE_FIGURES = True
SHOW_FIGURES = False
STYLE = 'whitegrid'
FIGURE_SIZE = (12, 6)
LARGE_FIGURE_SIZE = (18, 6)
EXTRA_LARGE_FIGURE_SIZE = (24, 6)
HEATMAP_SIZE = (30, 18)
DPI = 300

# Define tab10 colors for consistent styling
TAB10_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Plot enable/disable flags
PLOT_1_ENABLED = False  # Anomalies per feature
PLOT_2_ENABLED = False  # Score histogram
PLOT_3_ENABLED = False  # Sequence distribution
PLOT_4_ENABLED = False  # Score vs Sequence
PLOT_5_ENABLED = False  # Score vs Datetime
PLOT_6_ENABLED = False  # Cluster correlation and distribution
PLOT_7_ENABLED = True  # Cluster Anomaly Proportion Stacked Bar Plot
PLOT_8_ENABLED = False  # Cluster Anomaly Proportion Stacked Bar Plot
PLOT_9_ENABLED = False  # Sequence Length vs. Anomaly Score Violin Plot
PLOT_10_ENABLED = False  # Cluster Pairwise Scatter Matrix
PLOT_11_ENABLED = False  # Performance metrics
PLOT_12_ENABLED = False  # Ratio of anomalies
PLOT_13_ENABLED = False  # True positives, false positives, false negatives
PLOT_14_ENABLED = False  # Correlation heatmap

# Setup
sns.set_style(STYLE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE

# Create all necessary folders
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(SUBFOLDER_ANOMALIES, exist_ok=True)
os.makedirs(SUBFOLDER_FREQUENCY, exist_ok=True)
os.makedirs(SUBFOLDER_SEQUENCES, exist_ok=True)
os.makedirs(SUBFOLDER_SEQUENCE_PLOT, exist_ok=True)
os.makedirs(SUBFOLDER_ANOMALY_PLOT, exist_ok=True)
os.makedirs(SUBFOLDER_ANOMALY_TIMESERIES, exist_ok=True)
os.makedirs(SUBFOLDER_ANOMALY_DENSITY, exist_ok=True)
os.makedirs(SUBFOLDER_ANOMALY_PROPORTION, exist_ok=True)
os.makedirs(SUBFOLDER_SEQUENCE_VIOLIN, exist_ok=True)
os.makedirs(SUBFOLDER_CLUSTER_PAIRPLOT, exist_ok=True)

# Load summary metrics
df_summary = pd.read_csv(RESULTS_SUMMARY_CSV).set_index('file')

# Load global data for Plot 1 and Plot 14
df_if = pd.read_csv(GLOBAL_CSV)
for col in ['anomaly', 'is_anomaly', 'sequence']:
    df_if[col] = df_if[col].astype(int)
df_if['cluster'] = df_if.get('cluster', 0)

# Define all files for Plots 2-6
cluster_files = [f for f in glob.glob(CLUSTER_FILES_PATH) if not f.endswith('_if.csv')]
all_files = [GLOBAL_CSV] + cluster_files

# Plot 1: Anomalies per feature
if PLOT_1_ENABLED:
    numeric_columns = df_if.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['anomaly', 'is_anomaly', 'sequence', 'cluster']
    features_to_plot = [col for col in numeric_columns if col not in exclude_cols]
    for feature in features_to_plot:
        plt.figure(figsize=LARGE_FIGURE_SIZE)
        sns.scatterplot(data=df_if, x='datetime', y=feature, hue='anomaly', palette={0: 'blue', 1: 'red'}, alpha=0.7)
        plt.title(f"Anomalías Reales vs Detectadas: {feature.upper()}")
        plt.xlabel("Datetime")
        plt.ylabel(feature.upper())
        plt.xticks(rotation=45)
        plt.legend(title='Anomaly', labels=['Real', 'Detectada'])
        if SAVE_FIGURES:
            plt.savefig(f"{SUBFOLDER_ANOMALIES}/{feature}.png", dpi=DPI, bbox_inches='tight')
        if SHOW_FIGURES:
            plt.show()
        plt.close()
        print(f" [ GRÁFICO 1 ] {feature}.png")

# Plot 2: Score histogram
if PLOT_2_ENABLED:
    for file in all_files:
        df_cluster = pd.read_csv(file)
        cluster_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
        plt.figure(figsize=FIGURE_SIZE)
        sns.histplot(df_cluster['anomaly_score'], bins=50, kde=True, color='red')
        plt.title(f"Distribución del Score de Anomalía - {cluster_name}")
        plt.xlabel("Score de Anomalía")
        plt.ylabel("Frecuencia")
        if SAVE_FIGURES:
            plt.savefig(os.path.join(SUBFOLDER_FREQUENCY, f"{cluster_name}_frecuencia.png"), dpi=DPI)
        plt.close()
        print(f" [ GRÁFICO 2 ] {cluster_name}_frecuencia.png")

# Plot 3: Sequence distribution
if PLOT_3_ENABLED:
    for file in all_files:
        df_cluster = pd.read_csv(file)
        cluster_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
        df_cluster['sequence'] = df_cluster.get('sequence', 0)
        df_sequences = df_cluster[df_cluster['sequence'] > 0].copy() if df_cluster['sequence'].sum() > 0 else df_cluster.copy()
        df_plot = df_sequences.groupby('datetime', as_index=False)['sequence'].sum()
        df_plot = df_plot if not df_plot.empty else pd.DataFrame({'datetime': ['no_data'], 'sequence': [0]})
        df_plot['datetime_cat'] = pd.Categorical(df_plot['datetime'], categories=df_plot['datetime'], ordered=True)
        plt.figure(figsize=EXTRA_LARGE_FIGURE_SIZE)
        sns.barplot(data=df_plot, x='datetime_cat', y='sequence', color='skyblue')
        plt.title(f"Distribución de Secuencias de Anomalía - {cluster_name}")
        plt.xlabel("Datetime")
        plt.ylabel("Tamaño de la Secuencia de Anomalía")
        max_labels = 20
        step = max(1, len(df_plot) // max_labels)
        plt.xticks(ticks=range(0, len(df_plot), step), labels=df_plot['datetime'][::step], rotation=45)
        if SAVE_FIGURES:
            plt.savefig(os.path.join(SUBFOLDER_SEQUENCES, f"{cluster_name}_secuencias.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 3 ] {cluster_name}_secuencias.png")

# Plots 4 & 5: Score vs Sequence and Score vs Datetime
if PLOT_4_ENABLED or PLOT_5_ENABLED:
    for file in all_files:
        df_cluster = pd.read_csv(file)
        cluster_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
        df_cluster['cluster'] = df_cluster.get('cluster', cluster_name)
        
        # Plot 4: Score vs Sequence
        if PLOT_4_ENABLED:
            plt.figure(figsize=FIGURE_SIZE)
            sns.scatterplot(data=df_cluster, x='sequence', y='anomaly_score', hue='cluster', palette='tab10', alpha=0.6, size='sequence', sizes=(20, 200))
            plt.title(f"Score de anomalías y longitud de secuencia - {cluster_name}")
            plt.xlabel("Tamaño Secuencia")
            plt.ylabel("Score de Anomalía")
            plt.legend(title='Cluster', loc='upper right')
            if SAVE_FIGURES:
                plt.savefig(os.path.join(SUBFOLDER_SEQUENCE_PLOT, f"{cluster_name}_sequencia.png"), dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f" [ GRÁFICO 4 ] {cluster_name}_sequencia.png")
        
        # Plot 5: Score vs Datetime with anomaly highlighting
        if PLOT_5_ENABLED:
            plt.figure(figsize=LARGE_FIGURE_SIZE)
            sns.scatterplot(data=df_cluster, x='datetime', y='anomaly_score', hue='cluster', palette='tab10', size='sequence', sizes=(20, 200), alpha=0.6)
            df_cluster['anomaly'] = df_cluster.get('anomaly', 0).astype(int)
            anomalies = df_cluster[df_cluster['anomaly'] == 1]
            if not anomalies.empty:
                # Scale sequence sizes for anomalies to a smaller range (10 to 100)
                sizes = np.interp(anomalies['sequence'], (df_cluster['sequence'].min(), df_cluster['sequence'].max()), (10, 100))
                plt.scatter(anomalies['datetime'], anomalies['anomaly_score'], c='red', s=sizes, marker='X', 
                            edgecolor='black', linewidth=0.3, label='Real Anomaly')
            plt.title(f"Distribución temporal de scores - {cluster_name}")
            plt.xlabel("Datetime")
            plt.ylabel("Score de Anomalía")
            plt.xticks(rotation=45)
            plt.legend(title='Cluster / Anomalía', bbox_to_anchor=(1.05, 1), loc='upper left')
            if SAVE_FIGURES:
                plt.savefig(os.path.join(SUBFOLDER_ANOMALY_PLOT, f"{cluster_name}_anomalia.png"), dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f" [ GRÁFICO 5 ] {cluster_name}_anomalia.png")

# Plot 6: Anomaly Score Time Series with Rolling Mean
# Muestra cómo varían los scores de anomalía (anomaly_score) a lo largo del tiempo (datetime) para cada clúster, con una media móvil (promedio de 7 días) para suavizar fluctuaciones y destacar tendencias. Las anomalías reales (anomaly == 1) se marcan con cruces rojas.
# Útil para identificar patrones temporales (por ejemplo, picos estacionales en calidad del agua o problemas operativos en pozos) y evaluar si las anomalías reales coinciden con scores altos.
if PLOT_6_ENABLED:
    for file in all_files:
        try:
            df_cluster = pd.read_csv(file)
            cluster_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
            # Ensure required columns
            df_cluster['anomaly'] = df_cluster.get('anomaly', 0).astype(int)
            df_cluster['anomaly_score'] = df_cluster.get('anomaly_score', 0).astype(float)
            df_cluster['cluster'] = df_cluster.get('cluster', cluster_name)
            if 'datetime' not in df_cluster.columns:
                print(f" [ GRÁFICO 6 ] Skipping {cluster_name}_anomaly_timeseries.png: 'datetime' column missing")
                continue
            if df_cluster['datetime'].isna().all():
                print(f" [ GRÁFICO 6 ] Skipping {cluster_name}_anomaly_timeseries.png: Invalid datetime data")
                continue

            plt.figure(figsize=LARGE_FIGURE_SIZE)
            # Plot anomaly_score time series and rolling mean per cluster
            for idx, cluster in enumerate(df_cluster['cluster'].unique()):
                df_subset = df_cluster[df_cluster['cluster'] == cluster].sort_values('datetime')
                if df_subset.empty:
                    continue
                # Compute 7-day rolling mean
                df_subset['rolling_mean'] = df_subset['anomaly_score'].rolling(window=7, min_periods=1).mean()
                color = TAB10_COLORS[idx % len(TAB10_COLORS)]
                sns.lineplot(data=df_subset, x='datetime', y='anomaly_score', label=f'{cluster} Score', color=color)
                sns.lineplot(data=df_subset, x='datetime', y='rolling_mean', label=f'{cluster} Rolling Mean', 
                             linestyle='--', color=color)
                # Plot true anomalies
                anomalies = df_subset[df_subset['anomaly'] == 1]
                if not anomalies.empty:
                    sizes = np.interp(anomalies['sequence'], 
                                    (df_subset['sequence'].min(), df_subset['sequence'].max()), (10, 100))
                    plt.scatter(anomalies['datetime'], anomalies['anomaly_score'], c='red', marker='X', 
                                s=sizes, edgecolor='black', linewidth=0.3, label=f'{cluster} True Anomalies')
            
            plt.title(f"Tendencias Temporales de Scores de Anomalía - {cluster_name}")
            plt.xlabel("Datetime")
            plt.ylabel("Score de Anomalía")
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            if SAVE_FIGURES:
                plt.savefig(os.path.join(SUBFOLDER_ANOMALY_TIMESERIES, f"{cluster_name}_anomaly_timeseries.png"), 
                            dpi=DPI, bbox_inches='tight')
            if SHOW_FIGURES:
                plt.show()
            plt.close()
            print(f" [ GRÁFICO 6 ] {cluster_name}_anomaly_timeseries.png")
        except Exception as e:
            print(f" [ GRÁFICO 6 ] Error processing {cluster_name}_anomaly_timeseries.png: {str(e)}")

# Plot 7: 



# Plot 8: Cluster Anomaly Proportion Stacked Bar Plot
# Compara la proporción de anomalías reales (anomaly == 1) versus no anomalías (anomaly == 0) en cada clúster, mostrando qué clústeres tienen mayor prevalencia de anomalías.
# Ayuda a priorizar clústeres para monitoreo (por ejemplo, si ubicacion_Beniopa tiene más anomalías que eficiencia_Estado_del_Pozo).
if PLOT_8_ENABLED:
    for file in all_files:
        try:
            df_cluster = pd.read_csv(file)
            cluster_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
            df_cluster['anomaly'] = df_cluster.get('anomaly', 0).astype(int)
            df_cluster['cluster'] = df_cluster.get('cluster', cluster_name)

            # Compute anomaly proportions
            anomaly_counts = df_cluster.groupby('cluster')['anomaly'].value_counts(normalize=True).unstack().fillna(0)
            if anomaly_counts.empty:
                print(f" [ GRÁFICO 8 ] Skipping {cluster_name}_anomaly_proportion.png: No anomaly data")
                continue
            anomaly_counts.columns = ['Non-Anomaly', 'Anomaly']
            plt.figure(figsize=FIGURE_SIZE)
            anomaly_counts.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff0000'], ax=plt.gca())
            plt.title(f"Proporción de Anomalías por Cluster - {cluster_name}")
            plt.xlabel("Cluster")
            plt.ylabel("Proporción")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Estado')
            plt.tight_layout()
            if SAVE_FIGURES:
                plt.savefig(os.path.join(SUBFOLDER_ANOMALY_PROPORTION, f"{cluster_name}_anomaly_proportion.png"), 
                            dpi=DPI, bbox_inches='tight')
            if SHOW_FIGURES:
                plt.show()
            plt.close()
            print(f" [ GRÁFICO 8 ] {cluster_name}_anomaly_proportion.png")
        except Exception as e:
            print(f" [ GRÁFICO 8 ] Error processing {cluster_name}_anomaly_proportion.png: {str(e)}")

# Plot 9: Sequence Length vs. Anomaly Score Violin Plot
# Muestra la distribución de scores de anomalía (anomaly_score) para diferentes rangos de longitud de secuencia (sequence, agrupada en bins: 0, 1–5, 6–10, >10). Las anomalías reales se resaltan con puntos rojos.
# Explora si secuencias más largas (por ejemplo, anomalías persistentes en medida_Hidraulica) tienen scores más altos o diferentes distribuciones.
if PLOT_9_ENABLED:
    for file in all_files:
        try:
            df_cluster = pd.read_csv(file)
            cluster_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
            df_cluster['anomaly'] = df_cluster.get('anomaly', 0).astype(int)
            df_cluster['anomaly_score'] = df_cluster.get('anomaly_score', 0).astype(float)
            df_cluster['sequence'] = df_cluster.get('sequence', 0)
            df_cluster['cluster'] = df_cluster.get('cluster', cluster_name)

            # Bin sequence lengths
            bins = [0, 1, 5, 10, float('inf')]
            labels = ['0', '1-5', '6-10', '>10']
            df_cluster['sequence_bin'] = pd.cut(df_cluster['sequence'], bins=bins, labels=labels, include_lowest=True)
            if df_cluster['sequence_bin'].isna().all():
                print(f" [ GRÁFICO 9 ] Skipping {cluster_name}_sequence_anomaly_violin.png: Invalid sequence data")
                continue
            plt.figure(figsize=LARGE_FIGURE_SIZE)
            sns.violinplot(data=df_cluster, x='sequence_bin', y='anomaly_score', hue='cluster', palette='tab10')
            # Overlay true anomalies
            anomalies = df_cluster[df_cluster['anomaly'] == 1]
            if not anomalies.empty:
                sns.scatterplot(data=anomalies, x='sequence_bin', y='anomaly_score', color='red', marker='o', 
                                s=50, label='True Anomalies')
            plt.title(f"Distribución de Scores por Longitud de Secuencia - {cluster_name}")
            plt.xlabel("Longitud de Secuencia")
            plt.ylabel("Score de Anomalía")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            if SAVE_FIGURES:
                plt.savefig(os.path.join(SUBFOLDER_SEQUENCE_VIOLIN, f"{cluster_name}_sequence_anomaly_violin.png"), 
                            dpi=DPI, bbox_inches='tight')
            if SHOW_FIGURES:
                plt.show()
            plt.close()
            print(f" [ GRÁFICO 9 ] {cluster_name}_sequence_anomaly_violin.png")
        except Exception as e:
            print(f" [ GRÁFICO 9 ] Error processing {cluster_name}_sequence_anomaly_violin.png: {str(e)}")

# Plot 10: 



# Plot 11: Performance metrics
if PLOT_11_ENABLED:
    metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'mcc']
    df_summary[metrics].plot(kind='bar', figsize=(16, 6))
    plt.title("Métricas de Rendimiento")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f"{RESULTS_FOLDER}/11_metricas.png", dpi=DPI)
    plt.close()
    print(f" [ GRÁFICO 11 ] 11_metricas.png")

# Plot 12: Ratio of anomalies
if PLOT_12_ENABLED:
    ratio_metrics = ['anomalies_real', 'anomalies_detected', 'detections_correct', 'total_coincidences']
    df_summary[ratio_metrics].plot(kind='bar', figsize=(16, 6))
    plt.title("Ratio de Anomalías Reales y Detectadas")
    plt.ylabel("Ratio")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f"{RESULTS_FOLDER}/12_anomalias_reales_detectadas.png", dpi=DPI)
    plt.close()
    print(f" [ GRÁFICO 12 ] 12_anomalias_reales_detectadas.png")

# Plot 13: True positives, false positives, false negatives
if PLOT_13_ENABLED:
    df_summary[['detections_correct', 'false_positives', 'false_negatives']].plot(
        kind='bar', figsize=(16, 6), color=['blue', 'green', 'red'])
    plt.title("Verdaderas positivas, falsos positivos y falsos negativos")
    plt.ylabel("Número")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f"{RESULTS_FOLDER}/13_anomalias_falsas_verdaderas.png", dpi=DPI)
    plt.close()
    print(f" [ GRÁFICO 13 ] 13_anomalias_falsas_verdaderas.png")

# Plot 14: Correlation heatmap
if PLOT_14_ENABLED:
    numeric_cols = df_if.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=HEATMAP_SIZE)
    sns.heatmap(df_if[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', 
                cbar=True, annot_kws={"size": 3}, linewidths=0.3, linecolor='white')
    plt.title("Correlation Matrix - All Numeric Features", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=4)
    plt.yticks(rotation=0, fontsize=4)
    if SAVE_FIGURES:
        plt.savefig(f"{RESULTS_FOLDER}/14_correlation_matrix.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f" [ GRÁFICO 14 ] 14_correlation_matrix.png")