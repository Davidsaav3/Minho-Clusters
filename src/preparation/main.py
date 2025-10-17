# IMPORTS
import subprocess
import sys
import logging
import os

# CREAR CARPETA SI NO EXISTE
os.makedirs('../../results/preparation', exist_ok=True)

# CONFIGURAR LOG PARA SOBRESCRIBIR
logging.basicConfig(
    filename='../../results/preparation/log.txt',  # Archivo de log
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

# LISTA DE SCRIPTS EN ORDEN
scripts = [
    '01_metrics.py',
    '02_nulls.py',
    '03_codification.py',
    '04_scale.py',
    '05_variance.py',
]

log_print("\n[ INICIO ]")

# EJECUTAR PIPELINE CON SALIDA EN TIEMPO REAL
for script in scripts:
    log_print(f"\n[ EJECUTANDO ] {script}\n")
    try:
        # EJECUTAR SCRIPT Y LEER SALIDA LÍNEA A LÍNEA
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # LEER SALIDA STANDARD
        for line in process.stdout:
            log_print(line.rstrip())

        # LEER SALIDA DE ERRORES
        for line in process.stderr:
            log_print(line.rstrip(), level='error')

        process.wait()
        if process.returncode != 0:
            log_print(f"[ ERROR ] {script} terminó con código {process.returncode}", level='error')

    except Exception as e:
        log_print(f"[ EXCEPCIÓN ] {script}: {e}", level='error')

log_print("\n[ FIN ]")
