# FILE: main.py
# THIS IS THE MAIN SCRIPT THAT ORCHESTRATES THE ENTIRE PROCESS BY CALLING ALL OTHER MODULES IN SEQUENCE.

import sys
import os

# ENSURE WE'RE IN THE SRC DIRECTORY AND FILES EXIST
required_files = ['data_preparation.py', 'preliminary_analysis.py', 'isolation_forest_global.py', 'continuity_analysis.py', 'clustering.py', 'isolation_forest_per_cluster.py']
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"ERROR: MISSING FILES: {missing_files}")
    sys.exit(1)

# RUN EACH MODULE IN ORDER USING EXEC WITH EXPLICIT UTF-8 ENCODING; ADDED PRINTS FOR DEBUGGING
print("RUNNING DATA_PREPARATION...")
try:
    exec(open('data_preparation.py', encoding='utf-8').read())
    print("DATA_PREPARATION COMPLETED.")
except Exception as e:
    print(f"ERROR IN DATA_PREPARATION: {e}")
    sys.exit(1)

print("RUNNING PRELIMINARY_ANALYSIS...")
try:
    exec(open('preliminary_analysis.py', encoding='utf-8').read())
    print("PRELIMINARY_ANALYSIS COMPLETED.")
except Exception as e:
    print(f"ERROR IN PRELIMINARY_ANALYSIS: {e}")
    sys.exit(1)

print("RUNNING ISOLATION_FOREST_GLOBAL...")
try:
    exec(open('isolation_forest_global.py', encoding='utf-8').read())
    print("ISOLATION_FOREST_GLOBAL COMPLETED.")
except Exception as e:
    print(f"ERROR IN ISOLATION_FOREST_GLOBAL: {e}")
    sys.exit(1)

print("RUNNING CONTINUITY_ANALYSIS...")
try:
    exec(open('continuity_analysis.py', encoding='utf-8').read())
    print("CONTINUITY_ANALYSIS COMPLETED.")
except Exception as e:
    print(f"ERROR IN CONTINUITY_ANALYSIS: {e}")
    sys.exit(1)

print("RUNNING CLUSTERING...")
try:
    exec(open('clustering.py', encoding='utf-8').read())
    print("CLUSTERING COMPLETED.")
except Exception as e:
    print(f"ERROR IN CLUSTERING: {e}")
    sys.exit(1)

print("RUNNING ISOLATION_FOREST_PER_CLUSTER...")
try:
    exec(open('isolation_forest_per_cluster.py', encoding='utf-8').read())
    print("ISOLATION_FOREST_PER_CLUSTER COMPLETED.")
except Exception as e:
    print(f"ERROR IN ISOLATION_FOREST_PER_CLUSTER: {e}")
    sys.exit(1)

# FINAL MESSAGE AFTER FULL PROCESS
print("PROCESO EXTENDIDO COMPLETADO: DATASET ANALIZADO CON CARACTERIZACIÓN DE ANOMALÍAS. TODOS LOS RESULTADOS GUARDADOS EN '../results/'.")