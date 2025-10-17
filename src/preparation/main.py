# IMPORTS
import subprocess  # EJECUTAR OTROS SCRIPTS
import sys          # RUTA DE EJECUTABLE PYTHON

# LISTA DE SCRIPTS EN ORDEN
scripts = [
    '01_metrics.py',
    '02_nulls.py',
    '03_codification.py',
    '04_scale.py',
    '05_variance.py',
]

# EJECUTAR PIPELINE
for script in scripts:
    print(f"\n[ EJECUTANDO ] {script}")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    
    print(result.stdout)  # MOSTRAR SALIDA
    if result.stderr:     # MOSTRAR ERRORES
        print("[ ERRORES DETECTADOS ]")
        print(result.stderr)
        break

print("\n[ FIN ] PIPELINE COMPLETO")
