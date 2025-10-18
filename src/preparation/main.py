import subprocess
import sys
import logging
import os

# CONFIGURACIÓN GENERAL 
RESULTS_DIR = '../../results/preparation'   # CARPETA PRINCIPAL DE RESULTADOS
LOG_FILE = os.path.join(RESULTS_DIR, 'log.txt')  # ARCHIVO DE LOG
OVERWRITE_LOG = True   # TRUE = SOBRESCRIBIR LOG, FALSE = AGREGAR
SHOW_STDOUT = True     # TRUE = MOSTRAR SALIDA POR PANTALLA
SHOW_STDERR = True     # TRUE = MOSTRAR ERRORES POR PANTALLA

# LISTA DE SCRIPTS Y ACTIVACIÓN INDIVIDUAL
SCRIPTS = [
    {'name': '01_metrics.py', 'active': True},        # CALCULA MÉTRICAS INICIALES
    {'name': '02_nulls.py', 'active': True},          # TRATAMIENTO DE NULOS
    {'name': '03_codification.py', 'active': True},   # CODIFICACIÓN DE VARIABLES
    {'name': '04_scale.py', 'active': True},          # ESCALADO DE DATOS
    {'name': '05_variance.py', 'active': True},       # SELECCIÓN POR VARIANZA
]

# CREAR CARPETA DE RESULTADOS
os.makedirs(RESULTS_DIR, exist_ok=True)  # CREAR CARPETA PRINCIPAL SI NO EXISTE

# CONFIGURAR LOG
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w' if OVERWRITE_LOG else 'a',  # SOBRESCRIBIR O AGREGAR LOG
    level=logging.INFO,
    format='%(message)s',                    # SOLO MENSAJE, SIN FECHA NI NIVEL
    encoding='utf-8'                          # SOPORTE PARA ACENTOS Y Ñ
)

# FUNCIÓN AUXILIAR PARA LOG + PANTALLA
def log_print(msg, level='info'):
    """IMPRIME MENSAJE EN PANTALLA Y LO GUARDA EN LOG"""
    if level == 'info':
        logging.info(msg)
        if SHOW_STDOUT:
            print(msg)
    elif level == 'error':
        logging.error(msg)
        if SHOW_STDERR:
            print(msg)

# EJECUCIÓN
log_print("\n[ INICIO ]")  # MENSAJE INICIAL

for script in SCRIPTS:
    if not script['active']:
        log_print(f"[ SKIP ] {script['name']} DESACTIVADO")  # OMITIR SCRIPT DESACTIVADO
        continue

    log_print(f"\n[ EJECUTANDO SCRIPT ] {script['name']}\n")
    try:
        # EJECUTAR SCRIPT Y CAPTURAR SALIDA EN TIEMPO REAL
        process = subprocess.Popen(
            [sys.executable, script['name']],  # PYTHON + SCRIPT
            stdout=subprocess.PIPE,           # CAPTURAR SALIDA STDOUT
            stderr=subprocess.PIPE,           # CAPTURAR ERRORES STDERR
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # LEER SALIDA STDOUT LÍNEA A LÍNEA
        for line in process.stdout:
            log_print(line.rstrip())

        # LEER SALIDA STDERR LÍNEA A LÍNEA
        for line in process.stderr:
            log_print(line.rstrip(), level='error')

        process.wait()  # ESPERAR FINALIZACIÓN
        if process.returncode != 0:
            log_print(f"[ ERROR ] {script['name']} TERMINÓ CON CÓDIGO {process.returncode}", level='error')

    except Exception as e:
        log_print(f"[ EXCEPCIÓN ] {script['name']}: {e}", level='error')

log_print("\n[ FIN ]")  
