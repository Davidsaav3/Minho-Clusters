import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# ================================
# CONFIGURATION
# ================================
IF_GLOBAL_CSV = '../../results/execution/if_global.csv'
IF_01_CSV = '../../results/execution/01_if.csv'
CLUSTER_FILES_PATH = '../../results/execution/cluster_*.csv'
RESULTS_SUMMARY_CSV = '../../results/execution/06_results.csv'
RESULTS_FOLDER = '../../results/execution/plots'
FEATURE_TO_PLOT = 'nivel_plaxiquet'
SAVE_FIGURES = True
SHOW_FIGURES = False
STYLE = 'whitegrid'

os.makedirs(RESULTS_FOLDER, exist_ok=True)
print(f"[ INFO ] CARPETA '{RESULTS_FOLDER}' CREADA SI NO EXISTÍA")

sns.set_style(STYLE)
plt.rcParams['figure.figsize'] = (12,6)

# ================================
# 1-11. Original IF and Cluster Plots
# ================================

# --- LOAD MAIN DATA ---
df_if = pd.read_csv(IF_GLOBAL_CSV)
df_if['anomaly'] = df_if['anomaly'].astype(int)
df_if['is_anomaly'] = df_if['is_anomaly'].astype(int)
df_if['sequence'] = df_if['sequence'].astype(int)
if 'cluster' not in df_if.columns:
    df_if['cluster'] = 0

df_if_sorted = df_if.sort_values(['cluster', 'datetime'])

# 1. Anomalies vs Real (Scatter)
plt.figure(figsize=(18,6))
sns.scatterplot(data=df_if, x='datetime', y=FEATURE_TO_PLOT, hue='anomaly',
                palette={0:'gray',1:'red'}, alpha=0.7)
plt.title(f"Anomalies Detected vs Real: {FEATURE_TO_PLOT.upper()}")
plt.xlabel("Datetime")
plt.ylabel(FEATURE_TO_PLOT.upper())
plt.xticks(rotation=45)
plt.legend(title='Anomaly', labels=['Normal','Detected'])
if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/01_anomalies_vs_real.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation Heatmap
numeric_cols = df_if.select_dtypes(include=['float64','int64']).columns
plt.figure(figsize=(30,18))
sns.heatmap(df_if[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm',
            cbar=True, annot_kws={"size":3}, linewidths=0.3, linecolor='white')
plt.title("Correlation Matrix - All Numeric Features", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=4)
plt.yticks(rotation=0, fontsize=4)
if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/02_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Histogram of Anomaly Scores
plt.figure()
sns.histplot(df_if['anomaly_score'], bins=50, kde=True, color='blue')
plt.title("Distribution of Anomaly Scores")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/03_anomaly_score_distribution.png", dpi=300)
plt.close()

# 4. Anomaly Sequence Lengths over Time
df_if_01 = pd.read_csv(IF_01_CSV)
if 'cluster' not in df_if_01.columns:
    df_if_01['cluster'] = 0
df_sequences = df_if_01[df_if_01['sequence'] > 0]

plt.figure(figsize=(18,6))
sns.scatterplot(data=df_sequences, x='datetime', y='sequence', hue='cluster',
                palette='tab10', size='sequence', sizes=(20,200), alpha=0.7)
plt.title("Temporal Distribution of Repeated Anomalies (Sequence Length)")
plt.xlabel("Datetime")
plt.ylabel("Anomaly Sequence Length")
plt.xticks(rotation=45)
plt.legend(title='Cluster', bbox_to_anchor=(1.05,1), loc='upper left')
if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/04_anomaly_sequence_over_time.png", dpi=300, bbox_inches='tight')
plt.close()

# --- LOAD CLUSTER FILES ---
cluster_files = glob.glob(CLUSTER_FILES_PATH)
dfs_clusters = []

for file in cluster_files:
    if not file.endswith('_if.csv'):
        dfc = pd.read_csv(file)
        if 'datetime' not in dfc.columns and 'timestamp' in dfc.columns:
            dfc['datetime'] = dfc['timestamp']
        if 'cluster' not in dfc.columns:
            dfc['cluster'] = 0
        dfs_clusters.append(dfc)

df_all_clusters = pd.concat(dfs_clusters, ignore_index=True)

# 5. Histogram anomaly_score per cluster
plt.figure(figsize=(12,6))
sns.scatterplot(
    data=df_all_clusters,
    x='sequence',
    y='anomaly_score',
    hue='cluster',
    palette='tab10',
    alpha=0.6,
    size='sequence',
    sizes=(20,200),
    legend=False  # Desactiva la leyenda automática
)
plt.title("Anomaly Score vs Sequence Length by Cluster")
plt.xlabel("Sequence Length")
plt.ylabel("Anomaly Score")
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/06_sequence_vs_score_by_cluster.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Scatter sequence vs anomaly_score by cluster
plt.figure(figsize=(12,6))
sns.scatterplot(data=df_all_clusters, x='sequence', y='anomaly_score', hue='cluster',
                palette='tab10', alpha=0.6, size='sequence', sizes=(20,200))
plt.title("Anomaly Score vs Sequence Length by Cluster")
plt.xlabel("Sequence Length")
plt.ylabel("Anomaly Score")
if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/06_sequence_vs_score_by_cluster.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Temporal Scatter of anomalies per cluster
plt.figure(figsize=(18,6))
sns.scatterplot(data=df_all_clusters, x='datetime', y='anomaly_score', hue='cluster',
                palette='tab10', size='sequence', sizes=(20,200), alpha=0.6)
plt.title("Temporal Distribution of Anomaly Scores per Cluster")
plt.xlabel("Datetime")
plt.ylabel("Anomaly Score")
plt.xticks(rotation=45)
plt.legend(title='Cluster', bbox_to_anchor=(1.05,1), loc='upper left')
if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/07_temporal_score_by_cluster.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# 8 Summary metrics from 06_results.csv
# ================================
df_summary = pd.read_csv(RESULTS_SUMMARY_CSV)
df_summary.set_index('file', inplace=True)

# 8. Performance Metrics
metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'mcc']
df_summary[metrics].plot(kind='bar', figsize=(16,6))
plt.title("Performance Metrics per Method / File")
plt.ylabel("Score")
plt.ylim(0,1.1)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metric')
plt.tight_layout()
if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/08_metrics_comparison.png", dpi=300)
plt.close()

# 9. Ratio detection vs False Positives
ratio_metrics = ['ratio_detection', 'ratio_fp']
df_summary[ratio_metrics].plot(kind='bar', figsize=(16,6), color=['green','red'])
plt.title("Ratio Detection vs False Positives")
plt.ylabel("Ratio")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/09_ratio_detection_fp.png", dpi=300)
plt.close()

# 10. Anomalies Detected vs Total Coincidences
df_summary[['anomalies_detected', 'total_coincidences']].plot(kind='bar', figsize=(16,6), color=['blue','gray'])
plt.title("Anomalies Detected vs Total Coincidences")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
if SAVE_FIGURES: plt.savefig(f"{RESULTS_FOLDER}/10_detected_vs_coincidences.png", dpi=300)
plt.close()

print("✅ All 10 visualizations saved in:", RESULTS_FOLDER)
