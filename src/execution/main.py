import subprocess  # EJECUTA OTROS SCRIPTS
import sys          # USA EL INTÉRPRETE DE PYTHON

# LISTA DE SCRIPTS
scripts = [
    '01_if.py',
    '02_manual_clustering.py',
    '03_automatic_clustering.py',
    '04_if_clusters.py',
    '05_temporal_continuity.py',
    '06_metrics.py'
]

# INICIO PIPELINE
print("[ INICIO ] PIPELINE DE ANOMALÍAS")

# EJECUCIÓN SECUENCIAL
for s in scripts:
    print(f"\n[ EJECUTANDO ] {s}")
    try:
        subprocess.run([sys.executable, f'{s}'], check=True)  # EJECUTA SCRIPT
    except subprocess.CalledProcessError as e:
        print(f"[ ERROR ] {s}: {e}")  # MUESTRA ERROR
        continue

# FIN PIPELINE
print("\n[ FIN ] PIPELINE COMPLETO")
