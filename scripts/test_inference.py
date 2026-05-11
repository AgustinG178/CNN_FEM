import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
import trimesh
import torch
from src.neural_manifold.inference import predict_volume_from_dicom
from src.tensor_pde.io_module import assemble_tensor_and_hu, extract_affine_matrix
import pydicom

import re

def obtener_modelos_disponibles(models_dir="data/03_models"):
    modelos = {}
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.startswith("unet_bone_topology_ep") and f.endswith(".pth"):
                match = re.search(r"ep(\d+)", f)
                if match:
                    ep_num = int(match.group(1))
                    modelos[ep_num] = os.path.join(models_dir, f)
    return modelos

def encontrar_dicom_valido(dicom_dir: str) -> str:
    for root, dirs, files in os.walk(dicom_dir):
        for f in files:
            if f.lower().endswith('.dcm'):
                fpath = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                    if hasattr(ds, 'ImageOrientationPatient'):
                        return fpath
                except: continue
    return None

import argparse

def main():
    parser = argparse.ArgumentParser(description="Inferencia Post-Entrenamiento (FEM)")
    parser.add_argument("--epoch", type=int, default=None, help="Número de época a procesar (Evita el menú interactivo para SLURM)")
    args = parser.parse_args()

    modelos_dict = obtener_modelos_disponibles()
    
    if not modelos_dict:
        print("[!] ERROR: No se encontraron modelos (.pth) en data/03_models/")
        return
        
    if args.epoch is not None:
        EPOCA = args.epoch
        if EPOCA not in modelos_dict:
            print(f"[!] ERROR: La época {EPOCA} pasada por argumento no existe.")
            return
    else:
        print("\n=============================================")
        print("        MODELOS DISPONIBLES PARA TEST        ")
        print("=============================================\n")
        
        epocas = sorted(modelos_dict.keys())
        columnas = 4
        
        for i in range(0, len(epocas), columnas):
            fila = epocas[i:i+columnas]
            print("  |  ".join([f"Época {ep:02d}" for ep in fila]))
            
        print("\n---------------------------------------------")
        print("  [ 00 ] -> Cancelar y salir del script")
        print("---------------------------------------------")
            
        try:
            entrada = input("\nIngresa el número de época que deseas probar: ").strip()
            
            if entrada in ['00', '0']:
                print("\n[!] Operación cancelada por el usuario. ¡Adiós!\n")
                return
                
            EPOCA = int(entrada)
            if EPOCA not in modelos_dict:
                print(f"[!] ERROR: La época {EPOCA} no existe en la carpeta.")
                return
        except ValueError:
            print("[!] ERROR: Debes ingresar un número entero válido.")
            return

    MODEL_PATH = modelos_dict[EPOCA]
    DIR_PACIENTE = "data/01_raw/Paciente_52"
    DIR_SALIDA = f"data/02_processed/test_ep{EPOCA}"
    
    os.makedirs(DIR_SALIDA, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n--- TEST DE INFERENCIA ÉPOCA {EPOCA} ({device.upper()}) ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[!] ERROR: No existe el modelo en {MODEL_PATH}")
        return

    # 1. Inferencia Pura (Confiamos en la red neuronal)
    print(f"-> Ejecutando inferencia sobre {DIR_PACIENTE}...")
    prob_volume = predict_volume_from_dicom(
        dicom_dir=DIR_PACIENTE,
        model_path=MODEL_PATH,
        device_str=device,
        return_probabilities=True
    )
    
    # 2. Generación Geométrica Delegada
    from src.fem_postprocessing.mesh_generation import process_and_save_dl_mesh
    print("\n-> Delegando reconstrucción geométrica al motor FEM Post-Processing...")
    process_and_save_dl_mesh(prob_volume, DIR_PACIENTE, DIR_SALIDA)
    
    out_file = os.path.join(DIR_SALIDA, "hueso_completo_FEM.stl")
    if os.path.exists(out_file):
        print(f"\n[✓] ¡ÉXITO! Modelo generado en: {out_file}")
        print(f"    Tamaño: {os.path.getsize(out_file)/(1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
