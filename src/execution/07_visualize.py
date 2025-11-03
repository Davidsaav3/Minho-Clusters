import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import matplotlib.patches as mpatches

# CONFIGURACIÓN GLOBAL
GLOBAL_CSV = '../../results/execution/04_global.csv'  # CSV GLOBAL PRINCIPAL
VARIANCE_CSV = '../../results/preparation/02_nulls.csv'  # CSV con datetime
CLUSTER_FILES_PATH = '../../results/execution/cluster_*.csv'  # PATRÓN ARCHIVOS DE CLUSTER
RESULTS_SUMMARY_CSV = '../../results/execution/06_results.csv'  # CSV RESUMEN RESULTADOS
RESULTS_FOLDER = '../../results/execution/plots'  # CARPETA RESULTADOS
SUBFOLDER_FREQUENCY = '../../results/execution/plots/0_frequencia'
ANOMALIES_DATASET_GLOBAL = '../../results/execution/plots/1_anomalies_dataset_global'
ANOMALIES_DATASET_SEQUENCE = '../../results/execution/plots/2_anomalies_dataset_sequence'
ANOMALIES_DATASET_SEQUENCE_2 = '../../results/execution/plots/02_anomalies_dataset_sequence'
ANOMALIES_IF_HORIZONTAL = '../../results/execution/plots/3_anomalies_if_horizontal_vertical'
ANOMALIES_IF_HORIZONTAL_2 = '../../results/execution/plots/03_anomalies_if_horizontal_vertical'
TYPE_HORIZONTAL_GROUPS = '../../results/execution/plots/4_type_horizontal_grups'
NEW_REGISTERS = '../../results/execution/plots/5_new_registers'
CONTAMINATION_CLUSTER_HORIZONTAL = '../../results/execution/plots/6_contamination_cluster_horizontal'
HORIZONTAL_CLUSTER = '../../results/execution/plots/7_horizontal_cluster'

SAVE_FIGURES = True
SHOW_FIGURES = False
STYLE = 'whitegrid'
FIGURE_SIZE = (12, 6)
LARGE_FIGURE_SIZE = (18, 6)
EXTRA_LARGE_FIGURE_SIZE = (24, 6)
HEATMAP_SIZE = (30, 18)
DPI = 300

# ACTIVACIÓN DE GRÁFICOS
PLOT_0_ENABLED = True
PLOT_1_ENABLED = True
PLOT_2_ENABLED = True
PLOT_02_ENABLED = True
PLOT_3_ENABLED = True
PLOT_03_ENABLED = True
PLOT_4_ENABLED = True
PLOT_5_ENABLED = True
PLOT_6_ENABLED = True
PLOT_7_ENABLED = True
PLOT_10_ENABLED = True
PLOT_11_ENABLED = True
PLOT_12_ENABLED = True
PLOT_13_ENABLED = True
PLOT_14_ENABLED = True


# CONFIGURAR ESTILO Y CREAR CARPETAS
sns.set_style(STYLE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE
for folder in [SUBFOLDER_FREQUENCY, ANOMALIES_DATASET_GLOBAL, ANOMALIES_DATASET_SEQUENCE,
               ANOMALIES_DATASET_SEQUENCE_2, ANOMALIES_IF_HORIZONTAL, ANOMALIES_IF_HORIZONTAL_2,
               TYPE_HORIZONTAL_GROUPS, NEW_REGISTERS, CONTAMINATION_CLUSTER_HORIZONTAL]:
    os.makedirs(folder, exist_ok=True)


# CARGAR DATOS BASE
df_summary = pd.read_csv(RESULTS_SUMMARY_CSV).set_index('file')  # CSV resumen
df_if = pd.read_csv(GLOBAL_CSV)  # CSV global
for col in ['anomaly','is_anomaly','sequence']:
    df_if[col] = df_if[col].astype(int)  # Asegurar enteros
df_if['cluster'] = df_if.get('cluster', 0)  # Asegurar columna cluster

# CARGAR datetime DESDE VARIANCE CSV
df_datetime = pd.read_csv(VARIANCE_CSV, usecols=['datetime']).reset_index(drop=True)
#
df_if = df_if.reset_index(drop=True)
df_if['datetime'] = df_datetime['datetime']  # Reemplazar datetime

# CARGAR datetime DESDE VARIANCE CSV
df_datetime = pd.read_csv(VARIANCE_CSV, usecols=['datetime']).reset_index(drop=True)

# OBTENER ARCHIVOS DE CLUSTER
cluster_files = [f for f in glob.glob(CLUSTER_FILES_PATH) if not f.endswith('_if.csv')]
all_files = [GLOBAL_CSV] + cluster_files

# ASIGNAR DATETIME A CADA CLUSTER
for file in all_files:
    df_cluster = pd.read_csv(file).reset_index(drop=True)
    
    # Solo asignar si el tamaño coincide
    if len(df_cluster) == len(df_datetime):
        df_cluster['datetime'] = df_datetime['datetime']
    else:
        print(f"⚠️ Tamaño diferente: {file}, datetime no asignado.")


# PLOT 0: COMBINADO HISTOGRAMA ORIGINAL Y NUEVO (SCORES POSITIVOS + KDE)
if PLOT_0_ENABLED:
    for file in cluster_files:
        df_cluster = pd.read_csv(file)
        for col in ['anomaly','is_anomaly','sequence']:
            df_cluster[col] = df_cluster[col].astype(int)
        cluster_name = os.path.splitext(os.path.basename(file))[0]

        # FILTRAR SOLO SCORES POSITIVOS PARA EL NUEVO GRÁFICO
        df_if_pos = df_if[df_if['anomaly_score'] > 0]
        df_cluster_pos = df_cluster[df_cluster['anomaly_score'] > 0]

        # CALCULAR MÉTRICAS NUMÉRICAS PARA APOYAR ANÁLISIS
        mean_global = df_if_pos['anomaly_score'].mean()
        mean_cluster = df_cluster_pos['anomaly_score'].mean()
        std_global = df_if_pos['anomaly_score'].std()
        std_cluster = df_cluster_pos['anomaly_score'].std()
        q25_global = df_if_pos['anomaly_score'].quantile(0.25)
        q75_global = df_if_pos['anomaly_score'].quantile(0.75)
        q25_cluster = df_cluster_pos['anomaly_score'].quantile(0.25)
        q75_cluster = df_cluster_pos['anomaly_score'].quantile(0.75)

        # MÉTRICAS GENUINE
        df_genuine_pos = df_if[df_if['genuine_anomaly_score'] > 0]
        mean_genuine = df_genuine_pos['genuine_anomaly_score'].mean()
        q25_genuine = df_genuine_pos['genuine_anomaly_score'].quantile(0.25)
        q75_genuine = df_genuine_pos['genuine_anomaly_score'].quantile(0.75)
        std_genuine = df_genuine_pos['anomaly_score'].std()

        # RATIO DE COINCIDENCIA DE ANOMALÍAS
        coinciden = ((df_cluster_pos['is_anomaly'] == 1) & (df_cluster_pos['anomaly'] == 1)).sum()
        total_cluster_anomalies = df_cluster_pos['is_anomaly'].sum()
        ratio_coincidencia = coinciden / total_cluster_anomalies if total_cluster_anomalies > 0 else 0

        # CREAR FIGURA CON 2 SUBPLOTS
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # IZQUIERDA: HISTOGRAMA ORIGINAL
        sns.histplot(df_if['anomaly_score'], bins=50, color='blue', alpha=0.4, label='Global', kde=False, ax=axes[0])
        sns.histplot(df_cluster['anomaly_score'], bins=50, color='red', alpha=0.6, label=cluster_name, kde=False, ax=axes[0])
        sns.histplot(df_if['genuine_anomaly_score'], bins=50, color='green', alpha=0.5, label='Genuine', kde=False, ax=axes[0])
        axes[0].set_title(f"Histograma Original - {cluster_name}")
        axes[0].set_xlabel("Score")
        axes[0].set_ylabel("Frecuencia")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # DERECHA: KDE = Kernel Density Estimate
        # Estima la densidad de probabilidad de los datos de forma continua usando “kernels” (normalmente gaussiano).
        # Una curva suave que representa la probabilidad de que un valor aparezca.
        sns.kdeplot(df_if_pos['anomaly_score'], color='blue', label='Global', fill=True, alpha=0.2, ax=axes[1])
        sns.kdeplot(df_cluster_pos['anomaly_score'], color='red', label=cluster_name, fill=True, alpha=0.3, common_norm=False, ax=axes[1])
        sns.kdeplot(df_genuine_pos['genuine_anomaly_score'], color='green', label='Genuine', fill=True, alpha=0.25, common_norm=False, ax=axes[1])

        # Líneas de referencia: medias
        axes[1].axvline(mean_global, color='blue', linestyle='--', label='Media Global')
        axes[1].axvline(mean_cluster, color='red', linestyle='--', label=f'Media {cluster_name}')
        axes[1].axvline(mean_genuine, color='green', linestyle='--', label='Media Genuine')

        # Líneas de percentiles 25% y 75
        axes[1].axvline(q25_global, color='blue', linestyle=':', alpha=0.5)
        axes[1].axvline(q75_global, color='blue', linestyle=':', alpha=0.5)
        axes[1].axvline(q25_cluster, color='red', linestyle=':', alpha=0.5)
        axes[1].axvline(q75_cluster, color='red', linestyle=':', alpha=0.5)
        axes[1].axvline(q25_genuine, color='green', linestyle=':', alpha=0.5)
        axes[1].axvline(q75_genuine, color='green', linestyle=':', alpha=0.5)

        # Añadir recuadro con métricas
        info_text = (
            f"Media Global: {mean_global:.2f}\n"
            f"Media Cluster: {mean_cluster:.2f}\n"
            f"Media Genuine: {mean_genuine:.2f}\n"
            f"\n"
            f"Std Global: {std_global:.2f}\n"
            f"Std Cluster: {std_cluster:.2f}\n"
            f"Std Genuine: {std_genuine:.2f}\n"
            f"\n"
            f"Percentiles Global 25%: {q25_global:.2f}\n"
            f"Percentiles Cluster 25%: {q25_cluster:.2f}\n"
            f"Percentiles Genuine 25%: {q25_genuine:.2f}\n"
            f"\n"
            f"Percentiles Global 75%: {q75_global:.2f}\n"
            f"Percentiles Cluster 75%: /{q75_cluster:.2f}\n"
            f"Percentiles Genuine 75%: {q75_genuine:.2f}"
        )
        axes[1].text(1.05, 0.5, info_text,
                     transform=axes[1].transAxes, ha='left', va='center', fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

        axes[1].set_title(f"Kernel Density Estimate (kdeplot) - {cluster_name}")
        axes[1].set_xlabel("Score de Anomalía")
        axes[1].set_ylabel("Densidad")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if SAVE_FIGURES:
            plt.savefig(os.path.join(SUBFOLDER_FREQUENCY, f"{cluster_name}_combined.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 0 ] {cluster_name}_frequencia.png")


# PLOT 1: ANOMALIES_DATASET_GLOBAL 
# Comparar detección de anomalías entre dataset limpio y contaminado.
if PLOT_1_ENABLED:
    exclude_cols = ['anomaly','is_anomaly','sequence','cluster','datetime','date','time','year','month','day','hour','minute','weekday','day_of_year','week_of_year','working_day','season','holiday','weekend']
    exclude_cols += [c for c in df_if.columns if c.startswith(('openweather_','aemet_'))]  # EXCLUIR COLUMNAS NO NUMÉRICAS/EXTERNAS
    features = [c for c in df_if.select_dtypes(include=[np.number]).columns if c not in exclude_cols]  # SELECCIONAR FEATURES NUMÉRICOS

    # CONVERTIR datetime A TIPO DATETIME Y ORDENAR
    df_if['datetime'] = pd.to_datetime(df_if['datetime'], dayfirst=True)
    df_if = df_if.sort_values('datetime').reset_index(drop=True)
    
    # FILTROS DE ANOMALÍAS
    detectadas = df_if['anomaly'] == 1
    genuinas = df_if['genuine_anomaly'] == 1
    contaminadas = df_if['is_anomaly'] == 1
    counts = {
        'det': detectadas.sum(), 'gen': genuinas.sum(), 'cont': contaminadas.sum(),
        'det_gen': (detectadas & genuinas).sum(), 'det_cont': (detectadas & contaminadas).sum(),
        'gen_cont': (genuinas & contaminadas).sum(), 'all': (detectadas & genuinas & contaminadas).sum()
    }

    for f in features:
        plt.figure(figsize=LARGE_FIGURE_SIZE)

        # PUNTOS NORMALES
        plt.scatter(df_if[~detectadas & ~contaminadas]['datetime'],
                    df_if[~detectadas & ~contaminadas][f],
                    color='blue', alpha=0.5, s=20, label='Normal')

        # Genuine anomaly score en verde (solo para anomaly_score)
        if f == 'anomaly_score' and 'genuine_anomaly_score' in df_if.columns:
            plt.scatter(df_if['datetime'], df_if['genuine_anomaly_score'],
                        color='green', alpha=0.5, s=20, label='Genuine')

        # ANOMALÍAS DETECTADAS
        det = df_if[detectadas].copy()
        det[f] = pd.to_numeric(det[f], errors='coerce')
        if not det.empty:
            sizes = np.interp(det['sequence'], (det['sequence'].min(), det['sequence'].max()), (150,400)) if 'sequence' in det.columns else 200
            plt.scatter(det['datetime'], det[f], color='orange', s=sizes, alpha=0.3, label=f'Detectadas ({counts["det"]})')

        # ANOMALÍAS GENUINAS
        gen = df_if[genuinas].copy()
        gen[f] = pd.to_numeric(gen[f], errors='coerce')
        if not gen.empty:
            sizes = np.interp(gen['genuine_sequence'], (gen['genuine_sequence'].min(), gen['genuine_sequence'].max()), (150,400)) if 'genuine_sequence' in gen.columns else 200
            plt.scatter(gen['datetime'], gen[f], color='red', s=sizes, alpha=0.4, label=f'Genuinas ({counts["gen"]})')

        # ANOMALÍAS CONTAMINADAS (con jitter en Y)
        cont = df_if[contaminadas].copy()
        cont[f] = pd.to_numeric(cont[f], errors='coerce')
        if not cont.empty:
            jitter = np.random.uniform(-0.01, 0.01, size=len(cont))
            plt.scatter(cont['datetime'], cont[f] + jitter, color='white', edgecolor='black',
                        marker='X', s=80, alpha=1, label=f'Contaminadas ({counts["cont"]})')

        # INTERSECCIONES
        inter = f"Inter: D&G={counts['det_gen']}, D&C={counts['det_cont']}, G&C={counts['gen_cont']}, Todos={counts['all']}"

        # CONFIGURACIÓN DEL GRÁFICO
        plt.title(f"Anomalías: {f.upper()}")
        plt.xlabel("Datetime")
        plt.ylabel(f.upper())
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(31))
        plt.legend(title='Leyenda', loc='upper right')
        plt.annotate(inter, xy=(0.01,-0.15), xycoords='axes fraction', fontsize=10)
        plt.annotate("Tamaño = longitud secuencia", xy=(0.01,-0.20), xycoords='axes fraction', fontsize=10, color='orange')

        # GUARDAR FIGURA
        if SAVE_FIGURES:
            plt.savefig(f"{ANOMALIES_DATASET_GLOBAL}/{f}.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 1 ] {f}.png")


if PLOT_2_ENABLED:
    for file in all_files:
        df = pd.read_csv(file)
        name = '04_global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]

        # ASIGNAR CLUSTER GLOBAL Y ANOMALÍAS GENUINAS SEGÚN FILA
        if len(df) == len(df_if):
            df['cluster'] = df_if['cluster'].values
            if 'genuine_anomaly' in df_if.columns: df['genuine_anomaly'] = df_if['genuine_anomaly'].values
            if 'genuine_anomaly_score' in df_if.columns: df['genuine_anomaly_score'] = df_if['genuine_anomaly_score'].values
        else:
            df['cluster'] = name
            df['genuine_anomaly'] = df.get('genuine_anomaly', 0)

        # Asegurar columna sequence
        df['sequence'] = df.get('sequence', 0)

        # Filtrar secuencias detectadas
        seq = df[df['sequence'] > 0].copy() if df['sequence'].sum() > 0 else df.copy()

        # AGRUPAR POR DATETIME Y CLUSTER PARA BARRAS
        plot_df = seq.groupby(['datetime','cluster'], as_index=False)['sequence'].sum().sort_values('datetime')
        if plot_df.empty:
            plot_df = pd.DataFrame({'datetime':['no_data'],'cluster':['none'],'sequence':[0]})

        plt.figure(figsize=EXTRA_LARGE_FIGURE_SIZE)
        unique_clusters = plot_df['cluster'].unique()
        palette = sns.color_palette('tab10', n_colors=len(unique_clusters))

        # BARRAS POR CLUSTER
        for i, c in enumerate(unique_clusters):
            cluster_df = plot_df[plot_df['cluster'] == c]
            n_cont = (seq[seq['cluster']==c]['is_anomaly'] == 1).sum()
            n_det  = (seq[seq['cluster']==c]['sequence'] > 0).sum()
            n_gen  = (seq[seq['cluster']==c]['genuine_anomaly'] == 1).sum()
            label = f"{c} (C:{n_cont} D:{n_det} G:{n_gen})"

            sns.barplot(
                data=cluster_df, x='datetime', y='sequence',
                color=palette[i], label=label, dodge=True, edgecolor=None, width=0.5
            )

        # ANOMALÍAS GENUINAS
        gen = seq[seq['genuine_anomaly'] == 1]
        if not gen.empty:
            for _, r in gen.iterrows():
                if r['datetime'] in plot_df['datetime'].values:
                    x = list(plot_df['datetime']).index(r['datetime'])
                    plt.scatter(x, r['sequence'], color='red', s=70, alpha=1.0, edgecolors='black', linewidths=0.6, zorder=12, label='Genuina')

        # ANOMALÍAS CONTAMINADAS
        cont = seq[seq['is_anomaly'] == 1]
        if not cont.empty:
            for _, r in cont.iterrows():
                if r['datetime'] in plot_df['datetime'].values:
                    x = list(plot_df['datetime']).index(r['datetime'])
                    plt.scatter(x, r['sequence'], color='white', s=80, marker='X', edgecolors='black', linewidths=1.5, zorder=13, label='Contaminada')

        # AJUSTE ETIQUETAS X Y LEYENDA
        max_labels = 20
        step = max(1, len(plot_df['datetime'].unique()) // max_labels)
        plt.xticks(ticks=range(0, len(plot_df['datetime'].unique()), step),
                   labels=plot_df['datetime'].unique()[::step], rotation=45)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(31))
        plt.legend(by_label.values(), by_label.keys(), title="Cluster / Anomalía", bbox_to_anchor=(1.05,1), loc='upper left')

        plt.title(f"Secuencias por Cluster - {name}")
        plt.xlabel("Datetime")
        plt.ylabel("Tamaño Secuencia")
        plt.tight_layout()

        if SAVE_FIGURES:
            os.makedirs(ANOMALIES_DATASET_SEQUENCE, exist_ok=True)
            plt.savefig(os.path.join(ANOMALIES_DATASET_SEQUENCE, f"{name}.png"), dpi=DPI, bbox_inches='tight')

        plt.close()
        print(f" [ GRÁFICO 2 ] {name}.png")

if PLOT_02_ENABLED:
    for file in all_files:
        df = pd.read_csv(file)
        name = '04_global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]

        # ASIGNAR CLUSTER GLOBAL Y ANOMALÍAS GENUINAS SEGÚN FILA
        if len(df) == len(df_if):
            df['cluster'] = df_if['cluster'].values
            if 'genuine_anomaly' in df_if.columns: df['genuine_anomaly'] = df_if['genuine_anomaly'].values
            if 'genuine_anomaly_score' in df_if.columns: df['genuine_anomaly_score'] = df_if['genuine_anomaly_score'].values
        else:
            df['cluster'] = name
            df['genuine_anomaly'] = df.get('genuine_anomaly', 0)

        df['sequence'] = df.get('sequence', 0)

        # Filtrar secuencias detectadas
        anomaly_mask = (df['anomaly'].values == 1) & (df_if['anomaly'].values == 1)
        seq = df[anomaly_mask & (df['sequence'] > 0)].copy()

        # AGRUPAR POR DATETIME Y CLUSTER PARA BARRAS
        plot_df = seq.groupby(['datetime','cluster'], as_index=False)['sequence'].sum().sort_values('datetime')
        if plot_df.empty:
            plot_df = pd.DataFrame({'datetime':['no_data'],'cluster':['none'],'sequence':[0]})

        plt.figure(figsize=EXTRA_LARGE_FIGURE_SIZE)
        unique_clusters = plot_df['cluster'].unique()
        palette = sns.color_palette('tab10', n_colors=len(unique_clusters))

        # --- BARRAS MÁS ESTRECHAS ---
        bar_width = 0.5  # ajustar ancho
        for i, c in enumerate(unique_clusters):
            cluster_df = plot_df[plot_df['cluster'] == c]
            n_cont = (seq[seq['cluster']==c]['is_anomaly'] == 1).sum()
            n_det  = (seq[seq['cluster']==c]['sequence'] > 0).sum()
            n_gen  = (seq[seq['cluster']==c]['genuine_anomaly'] == 1).sum()
            label = f"{c} (C:{n_cont} D:{n_det} G:{n_gen})"

            sns.barplot(
                data=cluster_df, x='datetime', y='sequence',
                color=palette[i], label=label, dodge=True, edgecolor=None, width=bar_width
            )

        # ANOMALÍAS GENUINAS
        gen = seq[seq['genuine_anomaly'] == 1]
        if not gen.empty:
            for _, r in gen.iterrows():
                if r['datetime'] in plot_df['datetime'].values:
                    x = list(plot_df['datetime']).index(r['datetime'])
                    plt.scatter(x, r['sequence'], color='red', s=70, alpha=1.0,
                                edgecolors='black', linewidths=0.6, zorder=12, label='Genuina')

        # ANOMALÍAS CONTAMINADAS
        cont = seq[seq['is_anomaly'] == 1]
        if not cont.empty:
            for _, r in cont.iterrows():
                if r['datetime'] in plot_df['datetime'].values:
                    x = list(plot_df['datetime']).index(r['datetime'])
                    plt.scatter(x, r['sequence'], color='white', s=80, marker='X',
                                edgecolors='black', linewidths=1.5, zorder=13, label='Contaminada')

        # ANOMALÍAS QUE COINCIDEN CON GLOBAL (cruces negras)
        if name != '04_global' and 'anomaly' in df.columns and 'anomaly' in df_if.columns:
            common_mask = (df['anomaly'] == 1) & (df_if['anomaly'] == 1)
            df_common = df[common_mask]
            if not df_common.empty:
                for _, r in df_common.iterrows():
                    if r['datetime'] in plot_df['datetime'].values:
                        x = list(plot_df['datetime']).index(r['datetime'])
                        # Desplazar ligeramente la X negra para no superponerse con rojo
                        x_disp = x + 0.15  # ajustar desplazamiento
                        plt.scatter(x_disp, r['sequence'], color='black', s=80, marker='X',
                                    edgecolors='black', linewidths=1.5, zorder=14, label='Coincide con global')

        # AJUSTE ETIQUETAS X Y LEYENDA
        max_labels = 20
        step = max(1, len(plot_df['datetime'].unique()) // max_labels)
        plt.xticks(ticks=range(0, len(plot_df['datetime'].unique()), step),
                   labels=plot_df['datetime'].unique()[::step], rotation=45)

        # LEYENDA CON CONTEOS
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = {}
        if name != '04_global' and not df_common.empty:
            clusters_common = df_common['cluster'].value_counts().to_dict()
            for cl, count in clusters_common.items():
                by_label[f"Coincide con global [C:{count} en cluster {cl}]"] = plt.Line2D([0],[0], marker='X', color='black', markersize=6, linestyle='')

        if not gen.empty:
            by_label[f"Genuina [C:{len(gen)}]"] = plt.Line2D([0],[0], marker='o', color='red', markersize=6, linestyle='')
        if not cont.empty:
            by_label[f"Contaminada [C:{len(cont)}]"] = plt.Line2D([0],[0], marker='X', color='white', markeredgecolor='black', markersize=6, linestyle='')

        for i, c in enumerate(unique_clusters):
            cluster_df = plot_df[plot_df['cluster'] == c]
            n_cont = (seq[seq['cluster']==c]['is_anomaly'] == 1).sum()
            n_det  = (seq[seq['cluster']==c]['sequence'] > 0).sum()
            n_gen  = (seq[seq['cluster']==c]['genuine_anomaly'] == 1).sum()
            by_label[f"{c} (C:{n_cont} D:{n_det} G:{n_gen})"] = plt.Line2D([0],[0], marker='o', color=palette[i], markersize=6, linestyle='')

        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(31))
        plt.legend(by_label.values(), by_label.keys(), title="Cluster / Anomalía", bbox_to_anchor=(1.05,1), loc='upper left')

        plt.title(f"Secuencias por Cluster - {name}")
        plt.xlabel("Datetime")
        plt.ylabel("Tamaño Secuencia")
        plt.tight_layout()

        if SAVE_FIGURES:
            os.makedirs(ANOMALIES_DATASET_SEQUENCE_2, exist_ok=True)
            plt.savefig(os.path.join(ANOMALIES_DATASET_SEQUENCE_2, f"{name}.png"), dpi=DPI, bbox_inches='tight')

        plt.close()
        print(f" [ GRÁFICO 02 ] {name}.png")




# PLOT 3: ANOMALIES_IF_HORIZONTAL, df_if= cluster, genuine_anomaly, genuine_sequence
# Analizar relación entre Isolation Forest global y agrupaciones horizontales.
if PLOT_3_ENABLED: 
    for file in all_files:
        df = pd.read_csv(file)
        name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]
        if len(df) == len(df_if):
            df['cluster'] = df_if['cluster'].values
            if 'genuine_anomaly' in df_if.columns: df['genuine_anomaly'] = df_if['genuine_anomaly'].values
            if 'genuine_sequence' in df_if.columns: df['genuine_sequence'] = df_if['genuine_sequence'].values
        else:
            df['cluster'] = name; df['genuine_anomaly'] = 0; df['genuine_sequence'] = 0
        df['sequence'] = df.get('sequence', 0); df['anomaly_score'] = df.get('anomaly_score', 0)

        plt.figure(figsize=LARGE_FIGURE_SIZE)

        # PLOT PRINCIPAL POR CLUSTER (SIN LEYENDA AUTOMÁTICA)
        palette = sns.color_palette('tab10', n_colors=df['cluster'].nunique())
        for i, c in enumerate(sorted(df['cluster'].unique())):
            sub = df[df['cluster']==c]
            plt.scatter(sub['datetime'], sub['anomaly_score'], color=palette[i], alpha=0.6, s=20, label=str(c))

        # ANOMALÍAS CONTAMINADAS
        cont = df[df['is_anomaly']==1]
        if not cont.empty:
            plt.scatter(cont['datetime'], cont['anomaly_score'], facecolor='white', edgecolor='black', marker='X', s=20, linewidth=0.5, alpha=1.0, zorder=12, label='Contaminada')

        # ANOMALÍAS GENUINAS
        gen = df[df['genuine_anomaly']==1]
        if not gen.empty:
            sizes = np.interp(gen['genuine_sequence'], (gen['genuine_sequence'].min(), gen['genuine_sequence'].max()), (50,300))
            plt.scatter(gen['datetime'], gen['anomaly_score'], color='red', s=sizes, alpha=0.2, edgecolor='black', linewidth=0.5, zorder=13, label='Genuina')

        # LEYENDA MANUAL CON CONTEOS POR CLUSTER
        new_handles = []
        new_labels = []
        for i, c in enumerate(sorted(df['cluster'].unique())):
            sub = df[df['cluster']==c]
            cont_count = (sub['is_anomaly']==1).sum()
            det_count = (sub['anomaly']==1).sum() if 'anomaly' in sub.columns else 0
            gen_count = (sub['genuine_anomaly']==1).sum()
            new_handles.append(plt.Line2D([0],[0], marker='o', color=palette[i], label=c, markersize=6, linestyle=''))
            new_labels.append(f"{c} [C:{cont_count} D:{det_count} G:{gen_count}]")

        # Añadir leyenda de clusters + anomalías
        if not cont.empty:
            new_handles.append(plt.Line2D([0],[0], marker='X', color='white', markeredgecolor='black', markersize=6, linestyle=''))
            new_labels.append('Contaminada')
        if not gen.empty:
            new_handles.append(plt.Line2D([0],[0], marker='o', color='red', markersize=6, linestyle=''))
            new_labels.append('Genuina')

        plt.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        plt.title(f"Scores Temporales - {name}")
        plt.xlabel("Datetime"); plt.ylabel("Score"); plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(31))
        plt.legend(new_handles, new_labels, title='Cluster / Anomalía', bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        if SAVE_FIGURES: 
            plt.savefig(os.path.join(ANOMALIES_IF_HORIZONTAL, f"{name}.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 3 ] {name}.png")
if PLOT_03_ENABLED:
    for file in all_files:
        df = pd.read_csv(file)
        name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]

        # Copia para leyenda
        df_full = df.copy()

        if len(df) == len(df_if):
            df['cluster'] = df_if['cluster'].values
            df_full['cluster'] = df_if['cluster'].values
            if 'genuine_anomaly' in df_if.columns:
                df['genuine_anomaly'] = df_if['genuine_anomaly'].values
                df_full['genuine_anomaly'] = df_if['genuine_anomaly'].values
            if 'genuine_sequence' in df_if.columns:
                df['genuine_sequence'] = df_if['genuine_sequence'].values
                df_full['genuine_sequence'] = df_if['genuine_sequence'].values
        else:
            df['cluster'] = name; df['genuine_anomaly'] = 0; df['genuine_sequence'] = 0
            df_full['cluster'] = name; df_full['genuine_anomaly'] = 0; df_full['genuine_sequence'] = 0

        df['sequence'] = df.get('sequence', 0); df['anomaly_score'] = df.get('anomaly_score', 0)
        df_full['sequence'] = df_full.get('sequence', 0); df_full['anomaly_score'] = df_full.get('anomaly_score', 0)

        plt.figure(figsize=LARGE_FIGURE_SIZE)

        # --- Fondo: todos los puntos por cluster ---
        palette = sns.color_palette('tab10', n_colors=df_full['cluster'].nunique())
        for i, c in enumerate(sorted(df_full['cluster'].unique())):
            sub = df_full[df_full['cluster'] == c]
            plt.scatter(sub['datetime'], sub['anomaly_score'], color=palette[i], alpha=0.3, s=20, label=str(c))

        # --- Resaltar anomalías ---
        # 1. Coinciden con global
        if 'anomaly' in df.columns and 'anomaly' in df_if.columns and name != 'global':
            common_mask = (df['anomaly'] == 1) & (df_if['anomaly'] == 1)
            df_common = df[common_mask]
        else:
            df_common = df[df['is_anomaly'] == 1].copy() if 'is_anomaly' in df.columns else pd.DataFrame()

        # 2. Todas las anomalías del cluster que no coinciden con el global
        df_cluster_anom = df[(df['is_anomaly'] == 1) & (~df.index.isin(df_common.index))] if not df_common.empty else df[df['is_anomaly'] == 1]

        # Plot: anomalías comunes con marker distinto
        if not df_common.empty:
            plt.scatter(df_common['datetime'], df_common['anomaly_score'],
                        facecolor='black', edgecolor='black', marker='X', s=30,
                        linewidth=0.8, alpha=1.0, zorder=12, label='Coincide con global')

        # Plot: otras anomalías del cluster
        if not df_cluster_anom.empty:
            plt.scatter(df_cluster_anom['datetime'], df_cluster_anom['anomaly_score'],
                        facecolor='yellow', edgecolor='black', marker='o', s=30,
                        linewidth=0.5, alpha=0.7, zorder=11, label='Solo cluster')

        # --- Anomalías genuinas ---
        gen = df[df['genuine_anomaly'] == 1]
        if not gen.empty:
            sizes = np.interp(gen['genuine_sequence'],
                              (gen['genuine_sequence'].min(), gen['genuine_sequence'].max()),
                              (50, 300))
            plt.scatter(gen['datetime'], gen['anomaly_score'],
                        color='red', s=sizes, alpha=0.2, edgecolor='black', linewidth=0.5,
                        zorder=13, label='Genuina')

        # --- Leyenda ---
        new_handles = []
        new_labels = []

        # Leyenda por cluster con conteos
        for i, c in enumerate(sorted(df_full['cluster'].unique())):
            sub = df_full[df_full['cluster'] == c]
            cont_count = (sub['is_anomaly'] == 1).sum() if 'is_anomaly' in sub.columns else 0
            det_count = (sub['anomaly'] == 1).sum() if 'anomaly' in sub.columns else 0
            gen_count = (sub['genuine_anomaly'] == 1).sum()
            new_handles.append(plt.Line2D([0], [0], marker='o', color=palette[i],
                                          label=c, markersize=6, linestyle=''))
            new_labels.append(f"{c} [C:{cont_count} D:{det_count} G:{gen_count}]")

        # Leyenda anomalías coincidentes con el global por cluster
        if not df_common.empty:
            clusters_common = df_common['cluster'].value_counts().to_dict()
            for cl, count in clusters_common.items():
                new_handles.append(plt.Line2D([0], [0], marker='X', color='black',
                                              markeredgecolor='black', markersize=6, linestyle=''))
                new_labels.append(f"Coincide con global [C:{count} en cluster {cl}]")

        # Leyenda anomalías solo cluster
        if not df_cluster_anom.empty:
            new_handles.append(plt.Line2D([0], [0], marker='o', color='yellow',
                                          markeredgecolor='black', markersize=6, linestyle=''))
            new_labels.append(f"Solo cluster [C:{len(df_cluster_anom)}]")

        # Leyenda anomalías genuinas
        if not gen.empty:
            new_handles.append(plt.Line2D([0], [0], marker='o', color='red',
                                          markersize=6, linestyle=''))
            new_labels.append(f"Genuina [C:{len(gen)}]")

        plt.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        plt.title(f"Scores Temporales - {name}")
        plt.xlabel("Datetime"); plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(31))
        plt.legend(new_handles, new_labels, title='Cluster / Anomalía',
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if SAVE_FIGURES:
            plt.savefig(os.path.join(ANOMALIES_IF_HORIZONTAL_2, f"{name}.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 03 ] {name}.png")



# PLOT 4: TYPE_HORIZONTAL_GROUPS, df_if= todo
# Ver si distintas clusterizaciones del global con varios métodos y parámetros para localizar zonas con más anomalías.
if PLOT_4_ENABLED: 
    df_global = pd.read_csv('../../results/execution/04_global.csv')  # LEER GLOBAL
    import glob
    files = []
    for pattern in [
        '../../results/execution/03_global_birch_*.csv',
        '../../results/execution/03_global_dbscan_*.csv',
        '../../results/execution/03_global_kmeans_*.csv',
        '../../results/execution/03_global_minibatch_*.csv'
    ]:
        files.extend(glob.glob(pattern))

    for file in files:
        df = pd.read_csv(file)
        name = os.path.splitext(os.path.basename(file))[0]

        # PREPARAR DF PARA PLOT
        df_plot = df_global.copy()
        df_plot['cluster'] = name  # COPIAR Y ASIGNAR CLUSTER

        if 'cluster' in df.columns:
            df_plot = df_plot.merge(df[['datetime','cluster']], on='datetime', how='left', suffixes=('','_new'))
            if 'cluster_new' in df_plot.columns:
                df_plot['cluster'] = df_plot['cluster_new'].fillna(df_plot['cluster'])
                df_plot.drop(columns='cluster_new', inplace=True)  # ACTUALIZAR CLUSTER

        df_plot['sequence'] = df_plot.get('sequence', 0)
        df_plot['anomaly_score'] = df_plot.get('anomaly_score', 0)

        plt.figure(figsize=LARGE_FIGURE_SIZE)
        unique_clusters = df_plot['cluster'].unique()
        colors = sns.color_palette("tab10", len(unique_clusters))

        handles, labels = [], []

        for i, c in enumerate(unique_clusters):
            sub = df_plot[df_plot['cluster']==c]

            # CONTEOS POR CLUSTER
            cont_count = (sub['is_anomaly']==1).sum()
            det_count = (sub['anomaly']==1).sum() if 'anomaly' in sub.columns else 0
            gen_count = (sub['genuine_anomaly']==1).sum() if 'genuine_anomaly' in sub.columns else 0
            label = f"{c} [C:{cont_count} D:{det_count} G:{gen_count}]"

            # TAMAÑO DE PUNTOS: SOLO VARIAR POR SEQUENCE SI ES ANOMALÍA
            if cont_count + gen_count > 0:
                seq = sub['sequence']
                if seq.min() == seq.max():
                    sizes = np.full(len(seq), 50)
                else:
                    sizes = np.interp(seq, (seq.min(), seq.max()), (20,200))
            else:
                sizes = np.full(len(sub), 50)  # tamaño fijo para clusters normales

            sc = plt.scatter(
                sub['datetime'],
                sub['anomaly_score'],
                color=colors[i % len(colors)],
                s=sizes,
                alpha=0.6,
                label=label
            )
            handles.append(sc)
            labels.append(label)

        # ANOMALÍAS CONTAMINADAS
        cont = df_plot[df_plot['is_anomaly']==1]
        sc_cont = None
        if not cont.empty:
            sc_cont = plt.scatter(
                cont['datetime'],
                cont['anomaly_score'],
                facecolor='white',
                edgecolor='black',
                marker='X',
                s=40,
                linewidth=1.5,
                alpha=1,
                zorder=12,
                label='Contaminada'
            )
            handles.append(sc_cont)
            labels.append('Contaminada')

        # ANOMALÍAS GENUINAS
        gen = df_global[df_global['genuine_anomaly']==1]
        sc_gen = None
        if not gen.empty:
            if gen['genuine_sequence'].min() == gen['genuine_sequence'].max():
                sizes = np.full(len(gen), 100)
            else:
                sizes = np.interp(gen['genuine_sequence'], (gen['genuine_sequence'].min(), gen['genuine_sequence'].max()), (50,300))
            sc_gen = plt.scatter(
                gen['datetime'],
                gen['anomaly_score'],
                color='red',
                s=sizes,
                alpha=0.2,
                edgecolor='black',
                linewidth=0.5,
                zorder=13,
                label='Genuina'
            )
            handles.append(sc_gen)
            labels.append('Genuina')

        # AJUSTES DE LA FIGURA
        plt.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        plt.title(f"Distribución temporal de scores - {name}")
        plt.xlabel("Datetime")
        plt.ylabel("Score de Anomalía")
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(31))
        plt.legend(handles, labels, title='Cluster / Anomalía', bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()

        # GUARDAR FIGURA
        if SAVE_FIGURES:
            plt.savefig(os.path.join(TYPE_HORIZONTAL_GROUPS, f"{name}.png"), dpi=DPI, bbox_inches='tight')
        plt.close()

        print(f" [ GRÁFICO 4 ] {name}.png")


# PLOT 5: NEW_REGISTERS... todo
# Evaluar si las anomalías tienden a agruparse según el algoritmo y configuracion.
if PLOT_5_ENABLED:
    predictive_files = glob.glob('../../results/execution/03_global_predictive_*.csv')  # OBTENER ARCHIVOS PREDICTIVOS
    for file in predictive_files:
        df = pd.read_csv(file); name = os.path.splitext(os.path.basename(file))[0]
        df['datetime'] = df.get('datetime', 0); df['anomaly_score'] = df.get('anomaly_score', 0)
        df['cluster'] = df.get('cluster', name); df['predictive'] = df.get('predictive', '')  # ASEGURAR COLUMNAS

        unique_clusters = df['cluster'].unique(); palette = sns.color_palette('tab10', n_colors=len(unique_clusters))
        predictive_clusters = set()
        for p in df['predictive']: 
            if pd.notna(p) and p != '': predictive_clusters.update(p.split('-'))  # EXTRAER CLUSTERS PREDICTIVOS

        plt.figure(figsize=LARGE_FIGURE_SIZE)
        for i, c in enumerate(unique_clusters):
            sub = df[df['cluster'] == c]
            # CONTAR IS_ANOMALY EN EL CLUSTER
            n_anom = (sub['is_anomaly'] == 1).sum()
            label = f"{c} ({n_anom})"
            if str(c) in predictive_clusters:
                label += " (Predictive)"
            plt.scatter(sub['datetime'], sub['anomaly_score'], color=palette[i], alpha=0.6, label=label)  # SCATTER POR CLUSTER

        # ANOMALÍAS CONTAMINADAS
        cont = df[df['is_anomaly'] == 1]
        if not cont.empty:
            plt.scatter(cont['datetime'], cont['anomaly_score'], facecolor='white', edgecolor='black', marker='X', s=20, linewidth=0.5, alpha=1.0, zorder=12, label='Contaminada')
        
        plt.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        plt.title(f"Scores Predictivos - {name}"); plt.xlabel("Datetime"); plt.ylabel("Score"); plt.xticks(rotation=45)
        handles, labels = plt.gca().get_legend_handles_labels(); by_label = dict(zip(labels, handles))
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(31))
        plt.legend(by_label.values(), by_label.keys(), title='Cluster / Tipo', bbox_to_anchor=(1.05,1), loc='upper left'); plt.tight_layout()
        if SAVE_FIGURES: plt.savefig(os.path.join(NEW_REGISTERS, f"{name}_predictive.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 5 ] {name}_predictive.png")


if PLOT_6_ENABLED:
    all_files = [f for f in glob.glob('../../results/execution/contaminated_*.csv') if not f.endswith('_if.csv')]

    for file in all_files:
        df_if = pd.read_csv(file)

        # EXCLUIR COLUMNAS NO NUMÉRICAS/EXTERNAS
        exclude_cols = ['anomaly','is_anomaly','sequence','cluster','datetime','date','time',
                        'year','month','day','hour','minute','weekday','day_of_year',
                        'week_of_year','working_day','season','holiday','weekend']
        exclude_cols += [c for c in df_if.columns if c.startswith(('openweather_','aemet_'))]

        # SELECCIONAR FEATURES NUMÉRICOS
        features = [c for c in df_if.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        # FILTROS DE ANOMALÍAS
        detectadas = df_if['anomaly'] == 1
        contaminadas = df_if['is_anomaly'] == 1
        counts = {
            'det': detectadas.sum(),
            'cont': contaminadas.sum(),
            'det_cont': (detectadas & contaminadas).sum()
        }

        # SOLO PLOT para 'anomaly_score'
        f = 'anomaly_score'
        if f not in df_if.columns:
            continue

        plt.figure(figsize=LARGE_FIGURE_SIZE)

        # Colores según cluster
        clusters = sorted(df_if['cluster'].unique())
        palette = sns.color_palette('tab10', n_colors=len(clusters))
        cluster_colors = {c: palette[i % len(palette)] for i, c in enumerate(clusters)}

        # PUNTOS NORMALES por cluster
        normal = df_if[~detectadas & ~contaminadas]
        for c in clusters:
            sub = normal[normal['cluster'] == c]
            if not sub.empty:
                plt.scatter(sub['datetime'], sub[f],
                            color=cluster_colors[c],
                            alpha=0.5, s=20,
                            label=f'Cluster {c} (Normal)')

        # ANOMALÍAS DETECTADAS
        det = df_if[detectadas]
        if not det.empty:
            sizes = np.interp(det['sequence'], (det['sequence'].min(), det['sequence'].max()), (150,400)) \
                    if 'sequence' in det.columns else 200
            plt.scatter(det['datetime'], det[f],
                        color='orange', s=sizes, alpha=0.3,
                        label=f'Detectadas ({counts["det"]})')

        # ANOMALÍAS CONTAMINADAS
        cont = df_if[contaminadas]
        if not cont.empty:
            plt.scatter(cont['datetime'], cont[f],
                        color='blue', edgecolor='black', marker='X', s=80, alpha=1,
                        label=f'Contaminadas ({counts["cont"]})')

        # INTERSECCIONES
        inter = f"Inter: D&C={counts['det_cont']}"

        # CONFIGURACIÓN GRÁFICA
        plt.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        plt.title(f"Anomalías: {f.upper()}")
        plt.xlabel("Datetime")
        plt.ylabel(f.upper())
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(31))
        plt.legend(title='Leyenda', loc='upper right', fontsize=8)
        plt.annotate(inter, xy=(0.01,-0.15), xycoords='axes fraction', fontsize=10)
        plt.annotate("Tamaño = longitud secuencia", xy=(0.01,-0.20),
                     xycoords='axes fraction', fontsize=10, color='orange')

        # GUARDAR FIGURA
        if SAVE_FIGURES:
            filename_only = os.path.basename(file).replace('.csv', '.png')
            plt.savefig(f"{CONTAMINATION_CLUSTER_HORIZONTAL}/{filename_only}", dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 6 ] {filename_only}")


# PLOT 10: CANTIDAD DE ANOMALÍAS Y GENUINAS POR CLUSTER (mejorado con doble_anomaly)
if PLOT_10_ENABLED:
    try:
        df_global = pd.read_csv(GLOBAL_CSV)
        df_global['cluster'] = df_global['cluster'].astype(str)
        
        # Preparar dataframes
        count_anomaly = pd.DataFrame()
        count_genuine = pd.DataFrame()
        count_double = pd.DataFrame()
        files = [GLOBAL_CSV] + cluster_files
        
        for f in files:
            name = 'global' if f == GLOBAL_CSV else os.path.splitext(os.path.basename(f))[0]
            df = pd.read_csv(f)
            
            # Asegurarse de que las columnas existen
            df['anomaly'] = df.get('anomaly', pd.Series(0, index=df.index)).astype(int)
            df['genuine_anomaly'] = df.get('genuine_anomaly', pd.Series(0, index=df.index)).astype(int)
            
            if len(df) != len(df_global):
                print(f"[AVISO] {name} omitido")
                continue
            df['cluster'] = df_global['cluster']
            
            # Cantidades absolutas
            count_anomaly = pd.concat([count_anomaly, df.groupby('cluster')['anomaly'].sum().rename(name)], axis=1)
            count_genuine = pd.concat([count_genuine, df.groupby('cluster')['genuine_anomaly'].sum().rename(name)], axis=1)
            
            # Conteo de "doble_anomaly": anomaly=1 en el archivo y también en el cluster global
            double_anomaly = np.where((df['anomaly'] == 1) & (df_global['anomaly'] == 1), 1, 0)
            count_double = pd.concat([count_double, pd.Series(double_anomaly, index=df.index).groupby(df['cluster']).sum().rename(name)], axis=1)
        
        if count_anomaly.empty and count_genuine.empty and count_double.empty:
            print(" [ GRÁFICO 10 ] Sin datos")
        else:
            df_count_anomaly = count_anomaly.T.fillna(0)
            df_count_genuine = count_genuine.T.fillna(0)
            df_count_double = count_double.T.fillna(0)
            
            # Colores consistentes para los clusters
            colors = plt.cm.tab10.colors[:len(df_count_anomaly.columns)]
            
            # Crear figura con tres subplots
            fig, axs = plt.subplots(3, 1, figsize=(max(12, len(files)*2), 20), constrained_layout=True)
            
            # 1. Genuine anomalies
            bars_genuine = df_count_genuine.plot(kind='bar', color=colors, ax=axs[0])
            axs[0].set_title("Cantidad de Anomalías Genuinas por Cluster")
            axs[0].set_xlabel("Archivo")
            axs[0].set_ylabel("Cantidad de Anomalías Genuinas")
            axs[0].legend(title='Cluster', bbox_to_anchor=(1.05,1), loc='upper left', fontsize=10)
            axs[0].tick_params(axis='x', rotation=45)
            axs[0].grid(axis='y', linestyle='--', alpha=0.7)
            for i, container in enumerate(bars_genuine.containers):
                axs[0].bar_label(container, labels=[str(int(v)) for v in df_count_genuine.iloc[:, i]], label_type='edge', fontsize=9)
            
            # 2. Detected anomalies
            bars_anomaly = df_count_anomaly.plot(kind='bar', color=colors, ax=axs[1])
            axs[1].set_title("Cantidad de Anomalías Detectadas por Cluster")
            axs[1].set_xlabel("Archivo")
            axs[1].set_ylabel("Cantidad de Anomalías")
            axs[1].legend(title='Cluster', bbox_to_anchor=(1.05,1), loc='upper left', fontsize=10)
            axs[1].tick_params(axis='x', rotation=45)
            axs[1].grid(axis='y', linestyle='--', alpha=0.7)
            for i, container in enumerate(bars_anomaly.containers):
                axs[1].bar_label(container, labels=[str(int(v)) for v in df_count_anomaly.iloc[:, i]], label_type='edge', fontsize=9)
            
            # 3. Doble anomalies (archivo y cluster global = 1)
            bars_double = df_count_double.plot(kind='bar', color=colors, ax=axs[2])
            axs[2].set_title("Cantidad de Doble Anomalías por Cluster")
            axs[2].set_xlabel("Archivo")
            axs[2].set_ylabel("Cantidad de Doble Anomalías")
            axs[2].legend(title='Cluster', bbox_to_anchor=(1.05,1), loc='upper left', fontsize=10)
            axs[2].tick_params(axis='x', rotation=45)
            axs[2].grid(axis='y', linestyle='--', alpha=0.7)
            for i, container in enumerate(bars_double.containers):
                axs[2].bar_label(container, labels=[str(int(v)) for v in df_count_double.iloc[:, i]], label_type='edge', fontsize=9)
            
            # Guardar figura
            if SAVE_FIGURES:
                plt.savefig(os.path.join(RESULTS_FOLDER, "10_anomalias_cluster.png"), dpi=DPI, bbox_inches='tight')
            plt.close()
            print(" [ GRÁFICO 10 ] 10_anomalias_cluster.png")
            
    except Exception as e:
        print(f" [ GRÁFICO 10 ] Error: {e}")


# PLOT 11-13: MÉTRICAS COMBINADAS
if any([PLOT_11_ENABLED, PLOT_12_ENABLED, PLOT_13_ENABLED]):
    fig, axes = plt.subplots(3,1,figsize=(16,18))
    if PLOT_11_ENABLED:
        df_summary[['precision','recall','f1_score','accuracy','mcc']].plot(kind='bar', ax=axes[0])  # MÉTRICAS
        axes[0].set_title("Métricas"); axes[0].set_ylabel("Score"); axes[0].set_ylim(0,1.1); axes[0].tick_params(axis='x',rotation=45)
    if PLOT_12_ENABLED:
        df_summary[['anomalies_real','anomalies_detected','detections_correct','total_coincidences']].plot(kind='bar', ax=axes[1])  # RATIOS
        axes[1].set_title("Ratios"); axes[1].set_ylabel("Ratio"); axes[1].tick_params(axis='x',rotation=45)
    if PLOT_13_ENABLED:
        df_summary[['detections_correct','false_positives','false_negatives']].plot(kind='bar', ax=axes[2], color=['blue','green','red'])  # TP/FP/FN
        axes[2].set_title("TP/FP/FN"); axes[2].set_ylabel("Nº"); axes[2].tick_params(axis='x',rotation=45)
    plt.tight_layout()
    if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/11_metrics.png", dpi=DPI)
    plt.close()
    print(" [ GRÁFICO 11 ] 11_metrics.png")


# PLOT 14: MATRIZ DE CORRELACIÓN
if PLOT_14_ENABLED:
    cols = df_if.select_dtypes(include=['float64','int64']).columns  # COLUMNAS NUMÉRICAS
    plt.figure(figsize=HEATMAP_SIZE)
    sns.heatmap(df_if[cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True, annot_kws={"size":3}, linewidths=0.3, linecolor='white')  # HEATMAP CORRELACIÓN
    plt.title("Matriz de Correlación", fontsize=14); plt.xticks(rotation=45, ha='right', fontsize=4); 
    plt.yticks(rotation=0, fontsize=4)
    if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/14_correlation_matrix.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(" [ GRÁFICO 14 ] 14_correlation_matrix.png")