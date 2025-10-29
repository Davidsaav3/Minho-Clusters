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
SUBFOLDER_FREQUENCY = '../../results/execution/plots/00_frequencia'
ANOMALIES_DATASET_GLOBAL = '../../results/execution/plots/01_anomalies_dataset_global'
ANOMALIES_DATASET_SEQUENCE = '../../results/execution/plots/02_anomalies_dataset_sequence'
SCORE = '../../results/execution/plots/03_scores'
TYPE_HORIZONTAL_GROUPS = '../../results/execution/plots/04_type_horizontal_grups'
ANOMALIES_IF_HORIZONTAL = '../../results/execution/plots/05_anomalies_if_horizontal'
NEW_REGISTERS = '../../results/execution/plots/06_new_registers'
HORIZONTAL_VERTICAL = '../../results/execution/plots/07_horizontal_vertical'

SAVE_FIGURES = True
SHOW_FIGURES = False
STYLE = 'whitegrid'
FIGURE_SIZE = (12, 6)
LARGE_FIGURE_SIZE = (18, 6)
EXTRA_LARGE_FIGURE_SIZE = (24, 6)
HEATMAP_SIZE = (30, 18)
DPI = 300
TAB10_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

PLOT_0_ENABLED = False
PLOT_1_ENABLED = False
PLOT_2_ENABLED = False
PLOT_3_ENABLED = False
PLOT_4_ENABLED = True
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
os.makedirs(SUBFOLDER_FREQUENCY, exist_ok=True)
os.makedirs(ANOMALIES_DATASET_GLOBAL, exist_ok=True)
os.makedirs(ANOMALIES_DATASET_SEQUENCE, exist_ok=True)
os.makedirs(SCORE, exist_ok=True)
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

if PLOT_0_ENABLED:
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
        if SAVE_FIGURES: plt.savefig(os.path.join(SUBFOLDER_FREQUENCY, f"{cluster_name}.png"), dpi=DPI)
        plt.close()
        print(f" [ GRÁFICO 0 ] {cluster_name}.png")

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
    detectadas = df_if['anomaly'] == 1
    genuinas = df_if['genuine_anomaly'] == 1
    contaminadas = df_if['is_anomaly'] == 1  # antes 'reales'

    count_detectadas = detectadas.sum()
    count_genuinas = genuinas.sum()
    count_contaminadas = contaminadas.sum()

    count_detect_genu = (detectadas & genuinas).sum()
    count_detect_cont = (detectadas & contaminadas).sum()
    count_genu_cont = (genuinas & contaminadas).sum()
    count_all_three = (detectadas & genuinas & contaminadas).sum()

    for feature in features_to_plot:
        plt.figure(figsize=LARGE_FIGURE_SIZE)

        # Normales
        plt.scatter(df_if[~detectadas & ~contaminadas]['datetime'],
                    df_if[~detectadas & ~contaminadas][feature],
                    color='blue', alpha=0.5, s=20, label='Normal')

        # Detectadas
        anomalies = df_if[detectadas].copy()
        if not anomalies.empty:
            sizes = np.interp(
                anomalies['sequence'] if 'sequence' in df_if.columns else np.ones(len(anomalies)),
                (anomalies['sequence'].min() if 'sequence' in df_if.columns else 1, 
                 anomalies['sequence'].max() if 'sequence' in df_if.columns else 1), 
                (150, 400)
            )
            plt.scatter(anomalies['datetime'], anomalies[feature],
                        color='orange', s=sizes, alpha=0.3,
                        label=f'Detectadas ({count_detectadas})')

        # Genuinas (rojas, tamaño según genuine_sequence)
        genuine_anomalies = df_if[genuinas].copy()
        if not genuine_anomalies.empty:
            sizes_genuine = np.interp(
                genuine_anomalies['genuine_sequence'] if 'genuine_sequence' in df_if.columns else np.ones(len(genuine_anomalies)),
                (genuine_anomalies['genuine_sequence'].min() if 'genuine_sequence' in df_if.columns else 1, 
                 genuine_anomalies['genuine_sequence'].max() if 'genuine_sequence' in df_if.columns else 1), 
                (150, 400)
            )
            plt.scatter(genuine_anomalies['datetime'], genuine_anomalies[feature],
                        color='red', marker='o', s=sizes_genuine, alpha=0.4,
                        label=f'Genuinas ({count_genuinas})')

        # Contaminadas (blancas, sin transparencia)
        contaminadas_df = df_if[contaminadas].copy()
        if not contaminadas_df.empty:
            plt.scatter(contaminadas_df['datetime'], contaminadas_df[feature],
                        color='white', edgecolor='black', marker='X', s=80, alpha=1,
                        label=f'Contaminadas ({count_contaminadas})')

        # Agregar anotaciones de intersección en la leyenda
        inter_label = (
            f"Intersecciones: Detect&Genu={count_detect_genu}, "
            f"Detect&Cont={count_detect_cont}, Genu&Cont={count_genu_cont}, "
            f"Todas={count_all_three}"
        )
        plt.title(f"Anomalías Detectadas, Genuinas y Contaminadas: {feature.upper()}")
        plt.xlabel("Datetime")
        plt.ylabel(feature.upper())
        plt.xticks(rotation=45)
        plt.legend(title='Leyenda', loc='upper right')
        plt.annotate(inter_label, xy=(0.01, -0.15), xycoords='axes fraction', fontsize=10, color='black', ha='left')
        plt.annotate("Tamaño del punto = longitud de la secuencia", xy=(0.01, -0.20), xycoords='axes fraction', fontsize=10, color='orange', ha='left')

        if SAVE_FIGURES:
            plt.savefig(f"{ANOMALIES_DATASET_GLOBAL}/{feature}.png", dpi=DPI, bbox_inches='tight')
        if SHOW_FIGURES:
            plt.show()
        plt.close()
        print(f" [ GRÁFICO 1 ] {feature}.png")

if PLOT_2_ENABLED:
    for file in all_files:
        df_cluster = pd.read_csv(file)
        cluster_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
        if len(df_cluster) == len(df_if):
            df_cluster['cluster'] = df_if['cluster'].values
            # Si existe la columna genuine_anomaly en el global, la añadimos para pintar
            if 'genuine_anomaly' in df_if.columns:
                df_cluster['genuine_anomaly'] = df_if['genuine_anomaly'].values
            if 'genuine_anomaly_score' in df_if.columns:
                df_cluster['genuine_anomaly_score'] = df_if['genuine_anomaly_score'].values
        else:
            df_cluster['cluster'] = cluster_name
            if 'genuine_anomaly' not in df_cluster.columns:
                df_cluster['genuine_anomaly'] = 0  # Evitar errores si falta la columna
            if 'genuine_anomaly_score' not in df_cluster.columns:
                df_cluster['genuine_anomaly_score'] = 0

        df_cluster['sequence'] = df_cluster.get('sequence', 0)
        df_sequences = df_cluster[df_cluster['sequence'] > 0].copy() if df_cluster['sequence'].sum() > 0 else df_cluster.copy()

        # DataFrame para el gráfico
        df_plot = df_sequences.groupby(['datetime', 'cluster'], as_index=False)['sequence'].sum().sort_values('datetime')
        if df_plot.empty: 
            df_plot = pd.DataFrame({'datetime': ['no_data'], 'cluster': ['none'], 'sequence': [0]})

        plt.figure(figsize=EXTRA_LARGE_FIGURE_SIZE)

        # Barras por cluster
        sns.barplot(data=df_plot, x='datetime', y='sequence', hue='cluster', palette='tab10', dodge=True, edgecolor=None, width=2)

        # Marcar anomalías genuinas (rojas transparentes)
        genuine_anomalies = df_sequences[df_sequences['genuine_anomaly'] == 1]
        if not genuine_anomalies.empty:
            for idx, row in genuine_anomalies.iterrows():
                if row['datetime'] in df_plot['datetime'].values:
                    x_pos = list(df_plot['datetime']).index(row['datetime'])
                    plt.scatter(
                        x=x_pos,
                        y=row['sequence'],
                        color='red',
                        s=70,
                        alpha=1.0,          # transparencia
                        edgecolors='black',
                        linewidths=0.6,
                        zorder=12,
                        label='Genuina'
                    )

        # Marcar anomalías contaminadas (cruz blanca con borde negro)
        contaminated_anomalies = df_sequences[df_sequences['is_anomaly'] == 1]
        if not contaminated_anomalies.empty:
            for idx, row in contaminated_anomalies.iterrows():
                if row['datetime'] in df_plot['datetime'].values:
                    x_pos = list(df_plot['datetime']).index(row['datetime'])
                    plt.scatter(
                        x=x_pos,
                        y=row['sequence'],
                        color='white',
                        s=80,
                        alpha=1.0,
                        marker='X',         # cruces sólidas
                        edgecolors='black', # borde negro
                        linewidths=1.5,
                        zorder=13,
                        label='Contaminada'
                    )

        # Cálculo de recuentos
        total_registros = len(df_sequences)
        total_genuine = (df_sequences['genuine_anomaly'] == 1).sum()
        total_contaminated = (df_sequences['is_anomaly'] == 1).sum()
        total_intersection = ((df_sequences['genuine_anomaly'] == 1) & (df_sequences['is_anomaly'] == 1)).sum()

        # Título
        plt.title(f"Distribución de Secuencias de Anomalía por Cluster - {cluster_name}")
        plt.xlabel("Datetime")
        plt.ylabel("Tamaño Total de Secuencias")

        # Limitar número de labels
        max_labels = 20
        step = max(1, len(df_plot['datetime'].unique()) // max_labels)
        plt.xticks(ticks=range(0, len(df_plot['datetime'].unique()), step),
                   labels=df_plot['datetime'].unique()[::step], rotation=45)

        # Leyenda enriquecida
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # evita duplicados
        legend_text = (
            f"Registros: {total_registros}\n"
            f"Genuinas: {total_genuine}\n"
            f"Contaminadas: {total_contaminated}\n"
            f"Genuina ∩ Contaminada: {total_intersection}"
        )

        legend = plt.legend(by_label.values(), by_label.keys(),
                            title=legend_text,
                            bbox_to_anchor=(1.05, 1),
                            loc='upper left')
        plt.setp(legend.get_title(), fontsize='small')

        plt.tight_layout()
        if SAVE_FIGURES: 
            plt.savefig(os.path.join(ANOMALIES_DATASET_SEQUENCE, f"{cluster_name}.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 2 ] {cluster_name}.png")

if PLOT_3_ENABLED:
    for file in all_files:
        df_cluster = pd.read_csv(file)
        cluster_name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]

        # Asignar cluster y columnas necesarias
        if len(df_cluster) == len(df_if):
            df_cluster['cluster'] = df_if['cluster'].values
            if 'genuine_anomaly' in df_if.columns:
                df_cluster['genuine_anomaly'] = df_if['genuine_anomaly'].values
            if 'genuine_sequence' in df_if.columns:
                df_cluster['genuine_sequence'] = df_if['genuine_sequence'].values
        else:
            df_cluster['cluster'] = cluster_name
            if 'genuine_anomaly' not in df_cluster.columns:
                df_cluster['genuine_anomaly'] = 0
            if 'genuine_sequence' not in df_cluster.columns:
                df_cluster['genuine_sequence'] = 0

        df_cluster['sequence'] = df_cluster.get('sequence', 0)
        df_cluster['anomaly_score'] = df_cluster.get('anomaly_score', 0)

        plt.figure(figsize=LARGE_FIGURE_SIZE)

        # Scatter principal por cluster
        sns.scatterplot(
            data=df_cluster,
            x='datetime',
            y='anomaly_score',
            hue='cluster',
            palette='tab10',
            size='sequence',
            sizes=(20, 200),
            alpha=0.6
        )

        # Marcar anomalías contaminadas (reales) con cruz blanca y borde negro
        contaminated = df_cluster[df_cluster['is_anomaly'] == 1]
        if not contaminated.empty:
            plt.scatter(
                contaminated['datetime'],
                contaminated['anomaly_score'],
                facecolor='white',
                edgecolor='black',
                marker='X',
                s=40,
                linewidth=1.5,
                alpha=1.0,
                zorder=12,
                label='Contaminada'
            )

        # Marcar anomalías genuinas con círculo rojo alpha 0.2 y borde negro
        genuine = df_cluster[df_cluster['genuine_anomaly'] == 1]
        if not genuine.empty:
            sizes = np.interp(
                genuine['genuine_sequence'],
                (genuine['genuine_sequence'].min(), genuine['genuine_sequence'].max()),
                (50, 300)
            )
            plt.scatter(
                genuine['datetime'],
                genuine['anomaly_score'],
                color='red',
                s=sizes,
                alpha=0.2,
                edgecolor='black',
                linewidth=0.5,
                zorder=13,
                label='Genuina'
            )

        plt.title(f"Distribución temporal de scores - {cluster_name}")
        plt.xlabel("Datetime")
        plt.ylabel("Score de Anomalía")
        plt.xticks(rotation=45)

        # Leyenda
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title='Cluster / Anomalía', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        if SAVE_FIGURES:
            plt.savefig(os.path.join(SCORE, f"{cluster_name}.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 3 ] {cluster_name}.png")


if PLOT_4_ENABLED:
    # Cargar columnas base del global original
    df_global_base = pd.read_csv('../../results/execution/04_global.csv')
    all_files = glob.glob('../../results/execution/03_global_*.csv')  # todos los 03_global_*

    for file in all_files:
        df_cluster = pd.read_csv(file)
        cluster_name = os.path.splitext(os.path.basename(file))[0]

        # Crear df con todas las columnas de la base
        df_plot = df_global_base.copy()

        # Sobreescribir la columna cluster con la del archivo actual
        if 'cluster' in df_cluster.columns:
            df_plot['cluster'] = df_cluster['cluster'].values
        else:
            df_plot['cluster'] = cluster_name

        # Asegurar columnas necesarias para plotting
        df_plot['sequence'] = df_plot.get('sequence', 0)
        df_plot['anomaly_score'] = df_plot.get('anomaly_score', 0)

        plt.figure(figsize=LARGE_FIGURE_SIZE)

        # Scatter principal por cluster (guardamos handles para leyenda)
        scatter = sns.scatterplot(
            data=df_plot,
            x='datetime',
            y='anomaly_score',
            hue='cluster',
            palette='tab10',
            size='sequence',
            sizes=(20, 200),
            alpha=0.6
        )

        # Recuperar handles de seaborn para clusters
        cluster_handles, cluster_labels = scatter.get_legend_handles_labels()
        # Solo dejar los de los clusters (el legend de seaborn incluye "size" también)
        cluster_handles = cluster_handles[1:len(df_plot['cluster'].unique())+1]
        cluster_labels = cluster_labels[1:len(df_plot['cluster'].unique())+1]

        # Anomalías contaminadas
        contaminated = df_plot[df_plot['is_anomaly'] == 1]
        if not contaminated.empty:
            sc_cont = plt.scatter(
                contaminated['datetime'],
                contaminated['anomaly_score'],
                facecolor='white',
                edgecolor='black',
                marker='X',
                s=40,
                linewidth=1.5,
                alpha=1.0,
                zorder=12,
                label='Contaminada'
            )

        # Anomalías genuinas (solo del global original)
        genuine = df_global_base[df_global_base['genuine_anomaly'] == 1]
        if not genuine.empty:
            sizes = np.interp(
                genuine['genuine_sequence'],
                (genuine['genuine_sequence'].min(), genuine['genuine_sequence'].max()),
                (50, 300)
            )
            sc_genuine = plt.scatter(
                genuine['datetime'],
                genuine['anomaly_score'],
                color='red',
                s=sizes,
                alpha=0.2,
                edgecolor='black',
                linewidth=0.5,
                zorder=13,
                label='Genuina'
            )

        plt.title(f"Distribución temporal de scores - {cluster_name}")
        plt.xlabel("Datetime")
        plt.ylabel("Score de Anomalía")
        plt.xticks(rotation=45)

        # Leyenda combinada clusters + contaminadas + genuinas
        handles = cluster_handles.copy()
        labels = cluster_labels.copy()
        if not contaminated.empty:
            handles.append(sc_cont)
            labels.append('Contaminada')
        if not genuine.empty:
            handles.append(sc_genuine)
            labels.append('Genuina')

        plt.legend(handles, labels, title='Cluster / Anomalía', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        if SAVE_FIGURES:
            plt.savefig(os.path.join(TYPE_HORIZONTAL_GROUPS, f"{cluster_name}.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 4 ] {cluster_name}.png")



if PLOT_10_ENABLED:
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
        if df_resultados.empty: print(" [ GRÁFICO 10 ] No se pudieron calcular proporciones de anomalías.")
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
            if SAVE_FIGURES: plt.savefig(os.path.join(RESULTS_FOLDER, "10_anomalias_cluster.png"), dpi=DPI, bbox_inches='tight')
            plt.close()
            print(" [ GRÁFICO 10 ] 04_anomalias_cluster.png")
    except Exception as e: print(f" [ GRÁFICO 10 ] Error: {e}")

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
    if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/11_metrics.png", dpi=DPI)
    plt.close()
    print(f" [ GRÁFICO 11 ] 11_metrics.png")

if PLOT_14_ENABLED:
    numeric_cols = df_if.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=HEATMAP_SIZE)
    sns.heatmap(df_if[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True, annot_kws={"size": 3}, linewidths=0.3, linecolor='white')
    plt.title("Correlation Matrix - All Numeric Features", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=4)
    plt.yticks(rotation=0, fontsize=4)
    if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/14_correlation_matrix.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f" [ GRÁFICO 12 ] 12_correlation_matrix.png")