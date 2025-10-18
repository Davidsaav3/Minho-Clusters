import subprocess  # EJECUTAR OTROS SCRIPTS DESDE PYTHON
import sys          # PARA USAR EL INTERPRETE ACTUAL
import logging      # REGISTRAR MENSAJES EN ARCHIVO
import os           # CREAR CARPETAS

# PARÁMETROS CONFIGURABLES
RESULTS_FOLDER = '../../results/execution'  # CARPETA DONDE SE GUARDAN RESULTADOS Y LOGS
LOG_FILE = os.path.join(RESULTS_FOLDER, 'log.txt')  # ARCHIVO DE LOG
LOG_LEVEL = logging.INFO                           # NIVEL DE LOG: DEBUG, INFO, WARNING, ERROR
LOG_OVERWRITE = True                               # TRUE = SOBRESCRIBIR LOG CADA EJECUCIÓN
SCRIPTS = [   
    '00_ contaminate.py',                                     # LISTA DE SCRIPTS A EJECUTAR EN ORDEN
    '01_if.py',
    '02_manual_clustering.py',
    '03_automatic_clustering.py',
    '04_if_clusters.py',
    '05_temporal_continuity.py',
    '06_metrics.py',
    '07_visualize.py'
]
SHOW_OUTPUT = True  # TRUE = IMPRIMIR SALIDA EN PANTALLA

# CREAR CARPETA DE RESULTADOS
os.makedirs(RESULTS_FOLDER, exist_ok=True)  # CREAR CARPETA SI NO EXISTE

# CONFIGURAR LOG
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w' if LOG_OVERWRITE else 'a',  # SOBRESCRIBIR O ADJUNTAR
    level=LOG_LEVEL,
    format='%(message)s',  # SIN FECHA NI NIVEL
    encoding='utf-8'
)

# FUNCIÓN AUXILIAR PARA LOG + PANTALLA
def log_print(msg, level='info'):
    """REGISTRAR MENSAJE EN LOG Y OPCIONALMENTE IMPRIMIR EN PANTALLA"""
    if level == 'info':
        logging.info(msg)
        if SHOW_OUTPUT:
            print(msg)
    elif level == 'error':
        logging.error(msg)
        if SHOW_OUTPUT:
            print(msg)

# EJECUTAR SCRIPTS
log_print("[ INICIO ]")  # MARCAR INICIO

for script in SCRIPTS:
    log_print(f"\n[ EJECUTANDO ] {script}\n")
    try:
        process = subprocess.Popen(
            [sys.executable, script],  # USAR INTERPRETE ACTUAL
            stdout=subprocess.PIPE,    # CAPTURAR SALIDA
            stderr=subprocess.PIPE,    # CAPTURAR ERRORES
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # LEER SALIDA LÍNEA POR LÍNEA
        for line in process.stdout:
            log_print(line.rstrip())

        # LEER ERRORES LÍNEA POR LÍNEA
        for line in process.stderr:
            log_print(line.rstrip(), level='error')

        process.wait()  # ESPERAR FIN DEL PROCESO
        if process.returncode != 0:
            log_print(f"[ ERROR ] {script} terminó con código {process.returncode}", level='error')

    except Exception as e:
        log_print(f"[ EXCEPCIÓN ] {script}: {e}", level='error')

log_print("\n[ FIN ]")  # MARCAR FIN DEL PROCESO