"""
Generador de Modelos 3D Evolutivos por Época.
Genera UNA sola malla unificada (pelvis + fémures) por cada checkpoint.

Uso:
    python generar_stl_epocas.py
"""
import os
import numpy as np
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
import trimesh
from src.neural_manifold.inference import predict_volume_from_dicom
from src.tensor_pde.io_module import assemble_tensor_and_hu, extract_affine_matrix
import pydicom

# CONFIGURACIÓN
EPOCAS = [1, 4, 7, 9]

# Paciente de prueba
DIR_PACIENTE = "data/01_raw/Paciente_21"

# Directorio de salida
DIR_SALIDA_BASE = "data/02_processed/evolucion_3d"

# Filtro físico: HU > 200 selecciona hueso cortical y trabecular.
# (Antes era -200 que solo excluía aire, dejando pasar músculo, grasa, etc.)
HU_THRESHOLD = 200


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
    
    # 1. Cargar volumen HU UNA sola vez
    print("-> Cargando volumen HU original para filtrado físico...")
    X_hu = assemble_tensor_and_hu(DIR_PACIENTE)
    hu_mask = (X_hu > HU_THRESHOLD)
    print(f"   Filtro HU > {HU_THRESHOLD}: {int(np.sum(hu_mask))}/{hu_mask.size} vóxeles son tejido")
    
    # Spacing
    print("-> Extrayendo spacing del DICOM...")
    dcm_path = encontrar_dicom_valido(DIR_PACIENTE)
    T = extract_affine_matrix(dcm_path)
    spacing = tuple(np.linalg.norm(T[i, :3]) for i in range(3))
    print(f"   Spacing: {spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.3f} mm")
    
    for ep in EPOCAS:
        model_path = f"data/03_models/unet_bone_topology_ep{ep}.pth"
        
        if not os.path.exists(model_path):
            print(f"\n[!] No se encontró {model_path}. Saltando.")
            continue
        
        out_dir = os.path.join(DIR_SALIDA_BASE, f"epoca_{ep}")
        os.makedirs(out_dir, exist_ok=True)
        for old_file in [f for f in os.listdir(out_dir) if f.endswith('.stl')]:
            os.remove(os.path.join(out_dir, old_file))
        
        print(f"\n{'='*60}")
        print(f"  ÉPOCA {ep}")
        print(f"{'='*60}")
        
        # 2. Inferencia
        print(f"-> Inferencia volumétrica...")
        prob_volume = predict_volume_from_dicom(
            dicom_dir=DIR_PACIENTE,
            model_path=model_path,
            device_str='cpu',
            return_probabilities=True
        )
        
        # 3. Filtro HU
        print(f"-> Filtro físico HU > {HU_THRESHOLD}...")
        min_shape = tuple(min(a, b) for a, b in zip(prob_volume.shape, hu_mask.shape))
        prob_filtered = prob_volume[:min_shape[0], :min_shape[1], :min_shape[2]] * \
                        hu_mask[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # 4. Umbral automático (Otsu) sobre vóxeles no-cero
        nonzero_probs = prob_filtered[prob_filtered > 0.01]
        if len(nonzero_probs) > 100:
            tau = max(0.30, min(threshold_otsu(nonzero_probs), 0.80))
        else:
            tau = 0.50
        
        print(f"-> Binarización Otsu (τ = {tau:.3f})...")
        binary_mask = (prob_filtered > tau).astype(np.float32)
        total_bone = int(np.sum(binary_mask))
        print(f"   Vóxeles óseos: {total_bone}")
        
        if total_bone < 100:
            print(f"   [!] Muy pocos vóxeles. Saltando.")
            continue
        
        # 5. Marching Cubes
        print(f"-> Marching Cubes...")
        try:
            verts, faces, normals, _ = marching_cubes(
                volume=binary_mask, level=0.5,
                spacing=spacing, step_size=2, allow_degenerate=False
            )
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            if mesh.volume < 0:
                mesh.invert()
            
            # Selección por volumen (descarta láminas planas)
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                for c in components:
                    if c.volume < 0:
                        c.invert()
                mesh = max(components, key=lambda c: abs(c.volume))
                print(f"   {len(components)} fragmentos → mayor volumen: {abs(mesh.volume):.0f} mm³")
            
            # Suavizado
            trimesh.smoothing.filter_taubin(mesh, iterations=10)
            
            out_file = os.path.join(out_dir, f"hueso_completo_ep{ep}.stl")
            mesh.export(out_file)
            size_mb = os.path.getsize(out_file) / (1024*1024)
            bounds = mesh.bounds
            dims = bounds[1] - bounds[0]
            print(f"   [✓] {out_file}")
            print(f"       {size_mb:.1f} MB, {len(mesh.faces)} caras")
            print(f"       Dimensiones: {dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f} mm")
            
        except Exception as e:
            print(f"   [✗] Error: {e}")
    
    # Resumen
    print(f"\n{'='*60}")
    print(f"  RESUMEN")
    print(f"{'='*60}")
    for ep in EPOCAS:
        out_dir = os.path.join(DIR_SALIDA_BASE, f"epoca_{ep}")
        stl_file = os.path.join(out_dir, f"hueso_completo_ep{ep}.stl")
        if os.path.exists(stl_file):
            size_mb = os.path.getsize(stl_file) / (1024*1024)
            print(f"  Época {ep}: {size_mb:.1f} MB")
        else:
            print(f"  Época {ep}: No generado")
    
    print(f"\n¡Abrí los STL en 3D Viewer o MeshLab!")

if __name__ == "__main__":
    main()
