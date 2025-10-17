# src/main.py
import subprocess
import sys

scripts = [
    '01_if_global.py',
    '02_cluster_manual.py',
    '03_cluster_filas.py',
    '04_if_por_cluster.py',
    '05_continuidad_temporal.py',
    '06_comparacion_if.py'
]

print("=== INICIANDO PIPELINE DE ANOMALÍAS ===")

for s in scripts:
    print(f"\n--- Ejecutando {s} ---")
    try:
        subprocess.run([sys.executable, f'{s}'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR al ejecutar {s}: {e}")
        continue

print("\n=== PIPELINE COMPLETO ===")
print("Todos los resultados intermedios y finales están en '../results/'")
