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

EPOCA = 8
DIR_PACIENTE = "data/01_raw/Paciente_52"
DIR_SALIDA = "data/02_processed/test_ep8"
HU_THRESHOLD = 200 # Filtro para hueso real
MODEL_PATH = f"data/03_models/unet_bone_topology_ep{EPOCA}.pth"

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

def main():
    os.makedirs(DIR_SALIDA, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n--- TEST DE INFERENCIA ÉPOCA {EPOCA} ({device.upper()}) ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[!] ERROR: No existe el modelo en {MODEL_PATH}")
        return

    # 1. Inferencia
    print(f"-> Ejecutando inferencia sobre {DIR_PACIENTE}...")
    prob_volume = predict_volume_from_dicom(
        dicom_dir=DIR_PACIENTE,
        model_path=MODEL_PATH,
        device_str=device,
        return_probabilities=True
    )
    
    # 2. Uso de la probabilidad de la red (IA decide el umbral)
    print("-> Analizando histograma de probabilidades (IA buscando umbral óptimo)...")
    # Tomamos solo las probabilidades que no sean cero para el cálculo
    prob_samples = prob_volume[prob_volume > 0.05]
    if len(prob_samples) > 1000:
        tau = threshold_otsu(prob_samples)
        # Ponemos un límite de seguridad: nunca menor a 0.3 ni mayor a 0.8
        tau = max(0.3, min(tau, 0.8))
    else:
        tau = 0.5
        
    print(f"-> Umbral automático calculado por la IA: {tau:.3f}")
    
    # 3. Binarización inteligente
    binary_mask = (prob_volume > tau).astype(np.float32)
    
    # 4. Marching Cubes y STL (Optimizado)
    print("-> Generando Malla 3D...")
    dcm_path = encontrar_dicom_valido(DIR_PACIENTE)
    T = extract_affine_matrix(dcm_path)
    spacing = tuple(np.linalg.norm(T[i, :3]) for i in range(3))
    
    verts, faces, normals, _ = marching_cubes(
        volume=binary_mask, level=0.5, spacing=spacing, step_size=2
    )
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # 5. Limpieza de ruido (Componentes Conexos)
    print("-> Limpiando ruido (conservando solo componentes mayores)...")
    # Filtramos por volumen: nos quedamos con lo que sea mayor al 5% del componente más grande
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        max_vol = max(c.area for c in components) # Usamos área porque el volumen puede ser 0 en mallas no cerradas
        mesh = trimesh.util.concatenate([c for c in components if c.area > max_vol * 0.1])
        print(f"   Reducción de fragmentos: {len(components)} -> {len(mesh.faces)} caras finales")

    # 6. Suavizado y exportación
    print("-> Suavizado de Taubin...")
    trimesh.smoothing.filter_taubin(mesh, iterations=10)
    
    out_file = os.path.join(DIR_SALIDA, f"resultado_ep{EPOCA}_Paciente_52_limpio.stl")
    mesh.export(out_file)
    
    print(f"\n[✓] ¡ÉXITO! Modelo generado en: {out_file}")
    print(f"    Tamaño: {os.path.getsize(out_file)/(1024*1024):.1f} MB")
    print(f"    Vértices: {len(mesh.vertices)}")

if __name__ == "__main__":
    main()
