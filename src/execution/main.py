import subprocess
import sys
import logging
import os

# CREAR CARPETA DE RESULTADOS
os.makedirs('../../results/execution', exist_ok=True)

# CONFIGURAR LOG PARA SOBRESCRIBIR
logging.basicConfig(
    filename='../../results/execution/log.txt',  # Archivo de log
    filemode='w',           # SOBRESCRIBIR cada ejecución
    level=logging.INFO,
    format='%(message)s',   # SIN fecha ni nivel
    encoding='utf-8'        # Para acentos y ñ
)

# FUNCIÓN AUXILIAR PARA LOG + PANTALLA
def log_print(msg, level='info'):
    if level == 'info':
        logging.info(msg)  # Se guarda en log.txt
        print(msg)         # Se imprime en pantalla
    elif level == 'error':
        logging.error(msg)
        print(msg)

# LISTA DE SCRIPTS
scripts = [
    '01_if.py',
    '02_manual_clustering.py',
    '03_automatic_clustering.py',
    '04_if_clusters.py',
    '05_temporal_continuity.py',
    '06_metrics.py'
]

log_print("[ INICIO ]")

for s in scripts:
    log_print(f"\n[ EJECUTANDO ] {s}\n")
    try:
        # EJECUTAR SCRIPT Y CAPTURAR SALIDA EN TIEMPO REAL
        process = subprocess.Popen(
            [sys.executable, s],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # LEER SALIDA LÍNEA POR LÍNEA
        for line in process.stdout:
            line = line.rstrip()
            log_print(line)

        for line in process.stderr:
            line = line.rstrip()
            log_print(line, level='error')

        process.wait()
        if process.returncode != 0:
            log_print(f"[ ERROR ] {s} terminó con código {process.returncode}", level='error')

    except Exception as e:
        log_print(f"[ EXCEPCIÓN ] {s}: {e}", level='error')

log_print("\n[ FIN ]")
