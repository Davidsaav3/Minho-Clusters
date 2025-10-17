import pandas as pd  # MANEJO DE DATAFRAMES
import glob          # BUSCAR ARCHIVOS
import os            # RUTAS Y DIRECTORIOS

# CREAR CARPETA DE RESULTADOS SI NO EXISTE
os.makedirs('../../results', exist_ok=True)

# FUNCION AUXILIAR PARA SECUENCIAS
def secuencia_info(series):
    """Cuenta secuencias consecutivas de 1s y devuelve total y longitud máxima"""
    count_seq = 0
    max_len = 0
    current_len = 0
    for val in series:
        if val == 1:
            current_len += 1
            if current_len == 1:
                count_seq += 1  # nueva secuencia
            if current_len > max_len:
                max_len = current_len
        else:
            current_len = 0
    return count_seq, max_len

# ARCHIVOS A PROCESAR
files = glob.glob('../../results/execution/cluster-*.csv')

# PROCESAR CADA ARCHIVO
for file_path in files:
    df = pd.read_csv(file_path)  # LEER CSV

    # VERIFICAR COLUMNA DE ANOMALÍAS
    if 'anomaly' not in df.columns and 'anomaly_global' not in df.columns:
        print(f"[ SKIP ] No hay columna de anomalías, se salta")
        continue

    # SELECCIONAR COLUMNA DE ANOMALÍAS
    anomaly_col = 'anomaly' if 'anomaly' in df.columns else 'anomaly_global'

    # CALCULAR LONGITUD DE SECUENCIAS CONSECUTIVAS
    seq_len = []
    for val in df[anomaly_col]:
        seq_len.append(0)  # inicializa, se reemplaza después

    # USAR FUNCION AUXILIAR PARA INFO
    count_seq, max_len = secuencia_info(df[anomaly_col])
    df['sequence'] = 0  # inicializa columna
    current_len = 0
    for i, val in enumerate(df[anomaly_col]):
        if val == 1:
            current_len += 1
        else:
            current_len = 0
        df.at[i, 'sequence'] = current_len

    # GUARDAR RESULTADO
    df.to_csv(file_path, index=False)
    print(f"[ GUARDADO ] Secuencias añadidas en {file_path}")
    print(f"[ INFO ] Secuencias de anomalías: {count_seq}, longitud máxima: {max_len}")
