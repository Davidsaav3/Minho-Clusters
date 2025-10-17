# main.py
import subprocess
import sys

# Lista de scripts en orden
scripts = [
    '01_exploracion.py',
    '02_tratamiento_nulos.py',
    '03_codificacion.py',
    '04_escalado.py',
    '05_seleccion.py',
]

for script in scripts:
    print(f"\n=== EJECUTANDO {script} ===")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("ERRORES DETECTADOS:")
        print(result.stderr)
        break
print("\nPipeline completo. Todos los archivos intermedios generados.")
