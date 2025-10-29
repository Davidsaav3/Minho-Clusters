import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os, glob, numpy as np, matplotlib.patches as mpatches

# CONFIGURACIÓN GLOBAL
GLOBAL_CSV = '../../results/execution/04_global.csv'  # CSV GLOBAL PRINCIPAL
CLUSTER_FILES_PATH = '../../results/execution/cluster_*.csv'  # PATRÓN ARCHIVOS DE CLUSTER
RESULTS_SUMMARY_CSV = '../../results/execution/06_results.csv'  # CSV RESUMEN RESULTADOS
RESULTS_FOLDER = '../../results/execution/plots'  # CARPETA RESULTADOS
SUBFOLDER_FREQUENCY = '../../results/execution/plots/00_frequencia'  # SUBCARPETA FRECUENCIAS
ANOMALIES_DATASET_GLOBAL = '../../results/execution/plots/01_anomalies_dataset_global'  # ANOMALÍAS GLOBALES
ANOMALIES_DATASET_SEQUENCE = '../../results/execution/plots/02_anomalies_dataset_sequence'  # ANOMALÍAS POR SECUENCIA
ANOMALIES_IF_HORIZONTAL = '../../results/execution/plots/03_anomalies_if_horizontal_vertical'  # ANOMALÍAS HORIZONTAL/IF
TYPE_HORIZONTAL_GROUPS = '../../results/execution/plots/04_type_horizontal_grups'  # SCORES HORIZONTALES
NEW_REGISTERS = '../../results/execution/plots/05_new_registers'  # REGISTROS PREDICTIVOS

SAVE_FIGURES = True  # GUARDAR FIGURAS
SHOW_FIGURES = False  # MOSTRAR FIGURAS
STYLE = 'whitegrid'  # ESTILO SEABORN
FIGURE_SIZE = (12, 6)  # TAMAÑO FIGURA POR DEFECTO
LARGE_FIGURE_SIZE = (18, 6)  # TAMAÑO FIGURA GRANDE
EXTRA_LARGE_FIGURE_SIZE = (24, 6)  # TAMAÑO FIGURA EXTRA GRANDE
HEATMAP_SIZE = (30, 18)  # TAMAÑO HEATMAP
DPI = 300  # RESOLUCIÓN FIGURAS

# ACTIVACIÓN DE GRÁFICOS
PLOT_0_ENABLED = True
PLOT_1_ENABLED = True
PLOT_2_ENABLED = True
PLOT_3_ENABLED = True
PLOT_4_ENABLED = True
PLOT_5_ENABLED = True
PLOT_6_ENABLED = True
PLOT_7_ENABLED = True
PLOT_8_ENABLED = True
PLOT_9_ENABLED = True
PLOT_10_ENABLED = True
PLOT_11_ENABLED = True
PLOT_12_ENABLED = True
PLOT_13_ENABLED = True
PLOT_14_ENABLED = True

# CONFIGURAR ESTILO Y CREAR CARPETAS
sns.set_style(STYLE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE
os.makedirs(SUBFOLDER_FREQUENCY, exist_ok=True)
os.makedirs(ANOMALIES_DATASET_GLOBAL, exist_ok=True)
os.makedirs(ANOMALIES_DATASET_SEQUENCE, exist_ok=True)
os.makedirs(ANOMALIES_IF_HORIZONTAL, exist_ok=True)
os.makedirs(TYPE_HORIZONTAL_GROUPS, exist_ok=True)
os.makedirs(NEW_REGISTERS, exist_ok=True)

# CARGAR DATOS BASE
df_summary = pd.read_csv(RESULTS_SUMMARY_CSV).set_index('file')  # LEER CSV RESUMEN
df_if = pd.read_csv(GLOBAL_CSV)  # LEER CSV GLOBAL
for col in ['anomaly','is_anomaly','sequence']: df_if[col] = df_if[col].astype(int)  # CONVERTIR A ENTERO
df_if['cluster'] = df_if.get('cluster', 0)  # ASEGURAR COLUMNA CLUSTER

# OBTENER ARCHIVOS DE CLUSTER
cluster_files = [f for f in glob.glob(CLUSTER_FILES_PATH) if not f.endswith('_if.csv')]
all_files = [GLOBAL_CSV] + cluster_files  # LISTA TODOS ARCHIVOS

# PLOT 0: DISTRIBUCIÓN DE SCORES (HISTOGRAMA)
if PLOT_0_ENABLED:
    for file in cluster_files:
        df_cluster = pd.read_csv(file)
        for col in ['anomaly','is_anomaly','sequence']: df_cluster[col] = df_cluster[col].astype(int)
        cluster_name = os.path.splitext(os.path.basename(file))[0]
        plt.figure(figsize=FIGURE_SIZE)
        sns.histplot(df_if['anomaly_score'], bins=50, color='blue', alpha=0.4, label='Global', kde=False)  # HISTOGRAMA GLOBAL
        sns.histplot(df_cluster['anomaly_score'], bins=50, color='red', alpha=0.6, label=cluster_name, kde=False)  # HISTOGRAMA CLUSTER
        plt.title(f"Distribución del Score de Anomalía - {cluster_name}")
        plt.xlabel("Score"); plt.ylabel("Frecuencia"); plt.legend()
        if SAVE_FIGURES: plt.savefig(os.path.join(SUBFOLDER_FREQUENCY, f"{cluster_name}.png"), dpi=DPI)
        plt.close()
        print(f" [ GRÁFICO 0 ] {cluster_name}.png")

# PLOT 1: ANOMALÍAS POR CARACTERÍSTICA (SCATTER)
if PLOT_1_ENABLED:
    exclude_cols = ['anomaly','is_anomaly','sequence','cluster','datetime','date','time','year','month','day','hour','minute','weekday','day_of_year','week_of_year','working_day','season','holiday','weekend']
    exclude_cols += [c for c in df_if.columns if c.startswith(('openweather_','aemet_'))]  # EXCLUIR COLUMNAS NO NUMÉRICAS/EXTERNAS
    features = [c for c in df_if.select_dtypes(include=[np.number]).columns if c not in exclude_cols]  # SELECCIONAR FEATURES NUMÉRICOS
    
    # FILTROS DE ANOMALÍAS
    detectadas = df_if['anomaly'] == 1
    genuinas = df_if['genuine_anomaly'] == 1
    contaminadas = df_if['is_anomaly'] == 1
    counts = {
        'det': detectadas.sum(), 'gen': genuinas.sum(), 'cont': contaminadas.sum(),
        'det_gen': (detectadas & genuinas).sum(), 'det_cont': (detectadas & contaminadas).sum(),
        'gen_cont': (genuinas & contaminadas).sum(), 'all': (detectadas & genuinas & contaminadas).sum()
    }  # CONTAR ANOMALÍAS

    for f in features:
        plt.figure(figsize=LARGE_FIGURE_SIZE)
        plt.scatter(df_if[~detectadas & ~contaminadas]['datetime'], df_if[~detectadas & ~contaminadas][f], color='blue', alpha=0.5, s=20, label='Normal')  # PUNTOS NORMALES
        det = df_if[detectadas]
        if not det.empty:
            sizes = np.interp(det['sequence'], (det['sequence'].min(), det['sequence'].max()), (150,400)) if 'sequence' in det.columns else 200
            plt.scatter(det['datetime'], det[f], color='orange', s=sizes, alpha=0.3, label=f'Detectadas ({counts["det"]})')  # ANOMALÍAS DETECTADAS
        gen = df_if[genuinas]
        if not gen.empty:
            sizes = np.interp(gen['genuine_sequence'], (gen['genuine_sequence'].min(), gen['genuine_sequence'].max()), (150,400)) if 'genuine_sequence' in gen.columns else 200
            plt.scatter(gen['datetime'], gen[f], color='red', s=sizes, alpha=0.4, label=f'Genuinas ({counts["gen"]})')  # ANOMALÍAS GENUINAS
        cont = df_if[contaminadas]
        if not cont.empty:
            plt.scatter(cont['datetime'], cont[f], color='white', edgecolor='black', marker='X', s=80, alpha=1, label=f'Contaminadas ({counts["cont"]})')  # ANOMALÍAS CONTAMINADAS
        
        inter = f"Inter: D&G={counts['det_gen']}, D&C={counts['det_cont']}, G&C={counts['gen_cont']}, Todos={counts['all']}"  # INTERSECCIONES
        plt.title(f"Anomalías: {f.upper()}"); plt.xlabel("Datetime"); plt.ylabel(f.upper())
        plt.xticks(rotation=45); plt.legend(title='Leyenda', loc='upper right')
        plt.annotate(inter, xy=(0.01,-0.15), xycoords='axes fraction', fontsize=10)
        plt.annotate("Tamaño = longitud secuencia", xy=(0.01,-0.20), xycoords='axes fraction', fontsize=10, color='orange')
        if SAVE_FIGURES: plt.savefig(f"{ANOMALIES_DATASET_GLOBAL}/{f}.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 1 ] {f}.png")

# PLOT 2: SECUENCIAS POR CLUSTER + ANOMALÍAS REALES Y CONTAMINADAS
if PLOT_2_ENABLED:
    for file in all_files:
        df = pd.read_csv(file)  # LEER CSV
        name = 'global' if file == GLOBAL_CSV else os.path.splitext(os.path.basename(file))[0]  # NOMBRE ARCHIVO
        if len(df) == len(df_if):
            df['cluster'] = df_if['cluster'].values  # ASIGNAR CLUSTER GLOBAL
            if 'genuine_anomaly' in df_if.columns: df['genuine_anomaly'] = df_if['genuine_anomaly'].values
            if 'genuine_anomaly_score' in df_if.columns: df['genuine_anomaly_score'] = df_if['genuine_anomaly_score'].values
        else:
            df['cluster'] = name; df['genuine_anomaly'] = df.get('genuine_anomaly', 0)  # DEFAULTS
        df['sequence'] = df.get('sequence', 0)  # ASEGURAR COLUMNA SEQUENCE
        seq = df[df['sequence'] > 0].copy() if df['sequence'].sum() > 0 else df.copy()  # FILTRAR SECUENCIAS

        plot_df = seq.groupby(['datetime','cluster'], as_index=False)['sequence'].sum().sort_values('datetime')  # AGRUPAR POR CLUSTER
        if plot_df.empty: plot_df = pd.DataFrame({'datetime':['no_data'],'cluster':['none'],'sequence':[0]})  # SIN DATOS

        plt.figure(figsize=EXTRA_LARGE_FIGURE_SIZE)
        sns.barplot(data=plot_df, x='datetime', y='sequence', hue='cluster', palette='tab10', dodge=True, edgecolor=None, width=2)  # BARRAS SECUENCIA

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

        # RESUMEN ANOMALÍAS
        total, g, c, gc = len(seq), (seq['genuine_anomaly']==1).sum(), (seq['is_anomaly']==1).sum(), ((seq['genuine_anomaly']==1)&(seq['is_anomaly']==1)).sum()
        legend_text = f"Regs: {total}\nGen: {g}\nCont: {c}\nGen∩Cont: {gc}"

        # AJUSTE ETIQUETAS
        max_labels = 20; step = max(1, len(plot_df['datetime'].unique())//max_labels)
        plt.xticks(ticks=range(0,len(plot_df['datetime'].unique()),step), labels=plot_df['datetime'].unique()[::step], rotation=45)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title=legend_text, bbox_to_anchor=(1.05,1), loc='upper left')
        plt.title(f"Secuencias por Cluster - {name}"); plt.xlabel("Datetime"); plt.ylabel("Tamaño Secuencia"); plt.tight_layout()
        if SAVE_FIGURES: plt.savefig(os.path.join(ANOMALIES_DATASET_SEQUENCE, f"{name}.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 2 ] {name}.png")

# PLOT 3: SCORE VS DATETIME + ANOMALÍAS
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
        sns.scatterplot(data=df, x='datetime', y='anomaly_score', hue='cluster', palette='tab10', size='sequence', sizes=(20,200), alpha=0.6)  # SCATTER SCORE

        # ANOMALÍAS CONTAMINADAS
        cont = df[df['is_anomaly'] == 1]
        if not cont.empty:
            plt.scatter(cont['datetime'], cont['anomaly_score'], facecolor='white', edgecolor='black', marker='X', s=40, linewidth=1.5, alpha=1.0, zorder=12, label='Contaminada')

        # ANOMALÍAS GENUINAS
        gen = df[df['genuine_anomaly'] == 1]
        if not gen.empty:
            sizes = np.interp(gen['genuine_sequence'], (gen['genuine_sequence'].min(), gen['genuine_sequence'].max()), (50,300))
            plt.scatter(gen['datetime'], gen['anomaly_score'], color='red', s=sizes, alpha=0.2, edgecolor='black', linewidth=0.5, zorder=13, label='Genuina')

        plt.title(f"Scores Temporales - {name}"); plt.xlabel("Datetime"); plt.ylabel("Score"); plt.xticks(rotation=45)
        handles, labels = plt.gca().get_legend_handles_labels(); by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title='Cluster / Anomalía', bbox_to_anchor=(1.05,1), loc='upper left'); plt.tight_layout()
        if SAVE_FIGURES: plt.savefig(os.path.join(ANOMALIES_IF_HORIZONTAL, f"{name}.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 3 ] {name}.png")

# PLOT 4: SCORES POR GRUPO HORIZONTAL
if PLOT_4_ENABLED:
    df_global = pd.read_csv('../../results/execution/04_global.csv')  # LEER GLOBAL
    for file in glob.glob('../../results/execution/03_global_*.csv'):
        df = pd.read_csv(file); name = os.path.splitext(os.path.basename(file))[0]
        df_plot = df_global.copy(); df_plot['cluster'] = name  # COPIAR Y ASIGNAR CLUSTER
        if 'cluster' in df.columns:
            df_plot = df_plot.merge(df[['datetime','cluster']], on='datetime', how='left', suffixes=('','_new'))
            if 'cluster_new' in df_plot.columns:
                df_plot['cluster'] = df_plot['cluster_new'].fillna(df_plot['cluster']); df_plot.drop(columns='cluster_new', inplace=True)  # ACTUALIZAR CLUSTER
        df_plot['sequence'] = df_plot.get('sequence', 0); df_plot['anomaly_score'] = df_plot.get('anomaly_score', 0)

        plt.figure(figsize=LARGE_FIGURE_SIZE)
        s = sns.scatterplot(data=df_plot, x='datetime', y='anomaly_score', hue='cluster', palette='tab10', size='sequence', sizes=(20,200), alpha=0.6)
        h, l = s.get_legend_handles_labels(); h, l = h[1:len(df_plot['cluster'].unique())+1], l[1:len(df_plot['cluster'].unique())+1]  # LEYENDA CLUSTERS

        # ANOMALÍAS CONTAMINADAS
        cont = df_plot[df_plot['is_anomaly']==1]; sc_cont = plt.scatter(cont['datetime'], cont['anomaly_score'], facecolor='white', edgecolor='black', marker='X', s=40, linewidth=1.5, alpha=1, zorder=12, label='Contaminada') if not cont.empty else None

        # ANOMALÍAS GENUINAS
        gen = df_global[df_global['genuine_anomaly']==1]; sc_gen = None
        if not gen.empty:
            sizes = np.interp(gen['genuine_sequence'], (gen['genuine_sequence'].min(), gen['genuine_sequence'].max()), (50,300))
            sc_gen = plt.scatter(gen['datetime'], gen['anomaly_score'], color='red', s=sizes, alpha=0.2, edgecolor='black', linewidth=0.5, zorder=13, label='Genuina')

        handles, labels = h.copy(), l.copy()
        if sc_cont: handles.append(sc_cont); labels.append('Contaminada')
        if sc_gen: handles.append(sc_gen); labels.append('Genuina')

        plt.title(f"Distribución temporal de scores - {name}"); plt.xlabel("Datetime"); plt.ylabel("Score de Anomalía"); plt.xticks(rotation=45)
        plt.legend(handles, labels, title='Cluster / Anomalía', bbox_to_anchor=(1.05,1), loc='upper left'); plt.tight_layout()
        if SAVE_FIGURES: plt.savefig(os.path.join(TYPE_HORIZONTAL_GROUPS, f"{name}.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 4 ] {name}.png")

# PLOT 5: REGISTROS PREDICTIVOS
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
            label = f"{c} (Predictive)" if str(c) in predictive_clusters else str(c)
            plt.scatter(sub['datetime'], sub['anomaly_score'], color=palette[i], alpha=0.6, label=label)  # SCATTER POR CLUSTER

        # ANOMALÍAS CONTAMINADAS
        cont = df[df['is_anomaly'] == 1]
        if not cont.empty:
            plt.scatter(cont['datetime'], cont['anomaly_score'], facecolor='white', edgecolor='black', marker='X', s=20, linewidth=0.5, alpha=1.0, zorder=12, label='Contaminada')

        plt.title(f"Scores Predictivos - {name}"); plt.xlabel("Datetime"); plt.ylabel("Score"); plt.xticks(rotation=45)
        handles, labels = plt.gca().get_legend_handles_labels(); by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title='Cluster / Tipo', bbox_to_anchor=(1.05,1), loc='upper left'); plt.tight_layout()
        if SAVE_FIGURES: plt.savefig(os.path.join(NEW_REGISTERS, f"{name}_predictive.png"), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f" [ GRÁFICO 5 ] {name}_predictive.png")

# PLOT 10: PROPORCIÓN DE ANOMALÍAS POR CLUSTER
if PLOT_10_ENABLED:
    try:
        df_global = pd.read_csv(GLOBAL_CSV); df_global['cluster'] = df_global['cluster'].astype(str)  # LEER GLOBAL
        res = pd.DataFrame(); files = [GLOBAL_CSV] + cluster_files
        for f in files:
            name = 'global' if f == GLOBAL_CSV else os.path.splitext(os.path.basename(f))[0]
            df = pd.read_csv(f); df['anomaly'] = df['anomaly'].astype(int)
            if len(df) != len(df_global): print(f"[AVISO] {name} omitido"); continue  # OMITIR SI DIFERENTE LONGITUD
            df['cluster'] = df_global['cluster']
            res = pd.concat([res, df.groupby('cluster')['anomaly'].mean().multiply(100).rename(name)], axis=1)  # CALCULAR % ANOMALÍAS
        if res.empty: print(" [ GRÁFICO 10 ] Sin datos")
        else:
            df_plot = res.T.fillna(0)
            plt.figure(figsize=(20,8)); df_plot.plot(kind='bar', cmap='tab10')  # BARRAS %
            plt.title("Proporción Anomalías por Cluster"); plt.xlabel("Archivo"); plt.ylabel("% Anomalías")
            plt.legend(title='Cluster', bbox_to_anchor=(1.05,1), loc='upper left'); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
            if SAVE_FIGURES: plt.savefig(os.path.join(RESULTS_FOLDER, "10_anomalias_cluster.png"), dpi=DPI, bbox_inches='tight')
            plt.close()
            print(" [ GRÁFICO 10 ] 10_anomalias_cluster.png")
    except Exception as e: print(f" [ GRÁFICO 10 ] Error: {e}")

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
    plt.title("Matriz de Correlación", fontsize=14); plt.xticks(rotation=45, ha='right', fontsize=4); plt.yticks(rotation=0, fontsize=4)
    if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/14_correlation_matrix.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(" [ GRÁFICO 14 ] 14_correlation_matrix.png")
