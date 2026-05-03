"""
Generador de Modelos 3D Evolutivos por Época.
Genera UNA sola malla unificada (pelvis + fémures) por cada checkpoint,
sin intentar separar dominios anatómicos. Esto permite evaluar visualmente
cómo la IA aprende la topología ósea completa a lo largo del entrenamiento.

Uso:
    python generar_stl_epocas.py
"""
import os
import numpy as np
from skimage.measure import marching_cubes
import trimesh
from src.neural_manifold.inference import predict_volume_from_dicom
from src.tensor_pde.io_module import assemble_tensor_and_hu, extract_affine_matrix
import pydicom

# CONFIGURACIÓN
EPOCAS = [1, 4, 7]

# Paciente de prueba (recordar que estan listados en el txt dentro de "data/04_training_patches")
DIR_PACIENTE = "data/01_raw/Fantoma_Pelvis"

# Directorio de salida
DIR_SALIDA_BASE = "data/02_processed/evolucion_3d"

# Umbrales adaptativos por época
UMBRALES_POR_EPOCA = {
    1: 0.15,
    4: 0.30,
    7: 0.40,
}
UMBRAL_DEFAULT = 0.40

# Filtro físico: todo vóxel con HU < este valor se descarta (es aire)
HU_THRESHOLD = -200


def encontrar_dicom_valido(dicom_dir: str) -> str:
    """Busca recursivamente un DICOM válido (no localizer) para extraer el spacing."""
    for root, dirs, files in os.walk(dicom_dir):
        for f in files:
            if f.lower().endswith('.dcm'):
                fpath = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                    image_type = getattr(ds, 'ImageType', [])
                    if 'LOCALIZER' not in image_type and hasattr(ds, 'ImageOrientationPatient'):
                        return fpath
                except Exception:
                    continue
    raise ValueError("No se encontraron DICOMs válidos.")


def main():
    os.makedirs(DIR_SALIDA_BASE, exist_ok=True)
    
    # 1. Cargar volumen HU y spacing UNA sola vez
    print("-> Cargando volumen HU original para filtrado físico...")
    X_hu = assemble_tensor_and_hu(DIR_PACIENTE)
    hu_mask = (X_hu > HU_THRESHOLD)
    print(f"   Filtro HU > {HU_THRESHOLD}: {int(np.sum(hu_mask))}/{hu_mask.size} vóxeles son tejido")
    
    print("-> Extrayendo spacing del DICOM...")
    dcm_path = encontrar_dicom_valido(DIR_PACIENTE)
    T = extract_affine_matrix(dcm_path)
    spacing = tuple(np.linalg.norm(T[i, :3]) for i in range(3))
    print(f"   Spacing físico: {spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.3f} mm")
    
    for ep in EPOCAS:
        model_path = f"data/03_models/unet_bone_topology_ep{ep}.pth"
        
        if not os.path.exists(model_path):
            print(f"\n[!] No se encontró {model_path}. Saltando época {ep}.")
            continue
        
        out_dir = os.path.join(DIR_SALIDA_BASE, f"epoca_{ep}")
        os.makedirs(out_dir, exist_ok=True)
        
        # Limpiar STL anteriores para evitar conflictos
        for old_file in [f for f in os.listdir(out_dir) if f.endswith('.stl')]:
            os.remove(os.path.join(out_dir, old_file))
        
        print(f"\n{'='*60}")
        print(f"  GENERANDO MODELO 3D UNIFICADO - ÉPOCA {ep}")
        print(f"{'='*60}")
        
        # 2. Inferencia
        print(f"-> Paso 1/4: Inferencia volumétrica...")
        prob_volume = predict_volume_from_dicom(
            dicom_dir=DIR_PACIENTE,
            model_path=model_path,
            device_str='cpu',
            return_probabilities=True
        )
        
        # 3. Filtro HU
        print(f"-> Paso 2/4: Filtro físico HU > {HU_THRESHOLD}...")
        min_shape = tuple(min(a, b) for a, b in zip(prob_volume.shape, hu_mask.shape))
        prob_filtered = prob_volume[:min_shape[0], :min_shape[1], :min_shape[2]] * \
                        hu_mask[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        pre = int(np.sum(prob_volume > 0.5))
        post = int(np.sum(prob_filtered > 0.5))
        print(f"   Falsos positivos eliminados: {pre - post}")
        
        # 4. Binarización
        umbral = UMBRALES_POR_EPOCA.get(ep, UMBRAL_DEFAULT)
        print(f"-> Paso 3/4: Binarización con τ = {umbral}...")
        binary_mask = (prob_filtered > umbral).astype(np.float32)
        total_bone = int(np.sum(binary_mask))
        print(f"   Vóxeles óseos: {total_bone}")
        
        if total_bone < 100:
            print(f"   [!] Muy pocos vóxeles. Saltando esta época.")
            continue
        
        # 5. Marching Cubes directo → 1 sola malla
        print(f"-> Paso 4/4: Marching Cubes + limpieza (1 malla unificada)...")
        try:
            verts, faces, normals, _ = marching_cubes(
                volume=binary_mask,
                level=0.5,
                spacing=spacing,
                step_size=1,
                allow_degenerate=False
            )
            
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            if mesh.volume < 0:
                mesh.invert()
            
            # Quedarnos solo con la componente más grande (elimina ruido pequeño)
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                mesh = max(components, key=lambda c: len(c.faces))
                print(f"   Se encontraron {len(components)} fragmentos. Conservando el mayor ({len(mesh.faces)} caras).")
            
            # Suavizado ligero para que se vea mejor
            trimesh.smoothing.filter_taubin(mesh, iterations=10)
            
            out_file = os.path.join(out_dir, f"hueso_completo_ep{ep}.stl")
            mesh.export(out_file)
            size_mb = os.path.getsize(out_file) / (1024*1024)
            print(f"\n   [✓] Malla exportada: {out_file} ({size_mb:.1f} MB, {len(mesh.faces)} caras)")
            
        except Exception as e:
            print(f"\n   [✗] Error: {e}")
    
    # Resumen
    print(f"\n{'='*60}")
    print(f"  RESUMEN")
    print(f"{'='*60}")
    for ep in EPOCAS:
        out_dir = os.path.join(DIR_SALIDA_BASE, f"epoca_{ep}")
        stl_file = os.path.join(out_dir, f"hueso_completo_ep{ep}.stl")
        if os.path.exists(stl_file):
            size_mb = os.path.getsize(stl_file) / (1024*1024)
            print(f"  Época {ep}: hueso_completo_ep{ep}.stl ({size_mb:.1f} MB)")
        else:
            print(f"  Época {ep}: No generado")
    
    print(f"\n¡Abrí los STL en 3D Viewer o MeshLab para compararlos!")

if __name__ == "__main__":
    main()
