import os
import sys
import subprocess

def main():
    # Record ID de Zenodo para TotalSegmentator CT (v2.0.1)
    ZENODO_RECORD_ID = "10047292"
    
    # Rutas de descarga
    RAW_DIR = "data/05_totalsegmentator/raw/"
    os.makedirs(RAW_DIR, exist_ok=True)
    
    print("====================================================")
    print(f" Iniciando descarga de TotalSegmentator (Zenodo {ZENODO_RECORD_ID})")
    print("====================================================\n")
    
    # 1. Asegurar que zenodo_get esté instalado
    try:
        import zenodo_get
    except ImportError:
        print("[*] Instalando zenodo_get...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zenodo-get"])
    
    # 2. Descargar
    # El flag -o especifica el directorio de salida
    print(f"[*] Guardando archivos en: {RAW_DIR}")
    print("[*] Advertencia: Esto puede tomar horas dependiendo de la conexión.")
    
    cmd = f"zenodo_get {ZENODO_RECORD_ID} -o {RAW_DIR}"
    ret_code = os.system(cmd)
    
    if ret_code == 0:
        print("\n[✓] ¡Descarga de TotalSegmentator completada con éxito!")
    else:
        print(f"\n[!] ERROR: La descarga falló con código {ret_code}.")

if __name__ == "__main__":
    main()
