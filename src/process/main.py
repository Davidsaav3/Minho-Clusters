# FILE: main.py
# MAIN SCRIPT: ORCHESTRATES PROCESS BY RUNNING ALL MODULES IN SEQUENCE

import sys
import os

required_files = ['a_data_preparation.py', 'b_preliminary_analysis.py', 'c_isolation_forest_global.py', 'd_continuity_analysis.py', 'e_clustering.py', 'f_isolation_forest_per_cluster.py', 'g_visualize_anomalies.py', 'g_visualize_anomalies.py']
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"❌ {missing_files}")
    sys.exit(1)

print("[ 🌱 a_data_preparation ]")
try:
    #exec(open('a_data_preparation.py', encoding='utf-8').read())
    print("✅")
except Exception as e:
    print(f"❌: {e}")
    print(f"\n")
    sys.exit(1)

print("[ 📊 b_preliminary_analysis ]")
try:
    exec(open('b_preliminary_analysis.py', encoding='utf-8').read())
    print("✅")
except Exception as e:
    print(f"❌: {e}")
    print(f"\n")
    sys.exit(1)

print("[ 🌐 c_isolation_forest_global ]")
try:
    exec(open('c_isolation_forest_global.py', encoding='utf-8').read())
    print("✅")
except Exception as e:
    print(f"❌: {e}")
    print(f"\n")
    sys.exit(1)

print("[ ⏳ d_continuity_analysis ]")
try:
    exec(open('d_continuity_analysis.py', encoding='utf-8').read())
    print("✅")
except Exception as e:
    print(f"❌: {e}")
    print(f"\n")
    sys.exit(1)

print("[ 🧩 e_clustering ]")
try:
    exec(open('e_clustering.py', encoding='utf-8').read())
    print("✅")
except Exception as e:
    print(f"❌: {e}")
    print(f"\n")
    sys.exit(1)

print("[ 🔍 f_isolation_forest_per_cluster ]")
try:
    exec(open('f_isolation_forest_per_cluster.py', encoding='utf-8').read())
    print("✅")
except Exception as e:
    print(f"❌: {e}")
    print(f"\n")
    sys.exit(1)

print("[ 🎨 g_visualize_anomalies ]")
try:
    exec(open('g_visualize_anomalies.py', encoding='utf-8').read())
    print("✅")
except Exception as e:
    print(f"❌: {e}")
    print(f"\n")
    sys.exit(1)

print("[ 📈 h_anomaly_metrics ]")
try:
    exec(open('h_anomaly_metrics.py', encoding='utf-8').read())
    print("✅")
except Exception as e:
    print(f"❌ h_anomaly_metrics: {e}")
    print(f"\n")
    sys.exit(1)

print("[ 🎉 PROCESO COMPLETADO ]")