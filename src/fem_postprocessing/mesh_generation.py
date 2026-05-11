import numpy as np
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
import trimesh
import os
import pydicom
from src.fem_postprocessing.topology_repair import optimize_mesh_quality, seal_geometry
from src.tensor_pde.io_module import extract_affine_matrix, assemble_tensor_and_hu

def process_and_save_dl_mesh(prob_volume: np.ndarray, dicom_dir: str, out_dir: str) -> None:
    r"""
    Versión 'Cerebro Otsu': Sincronizada con el éxito de test_inference.py
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Binarización Directa (Confiamos en la IA V2)
    print("-> Binarizando salida de la red (Certeza > 40%)...")
    tau = 0.4
    binary_mask = (prob_volume > tau).astype(np.float32)
    
    dcm_path = None
    for root, dirs, files in os.walk(dicom_dir):
        for f in files:
            if f.lower().endswith('.dcm'):
                fpath = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                    if hasattr(ds, 'ImageOrientationPatient'):
                        dcm_path = fpath; break
                except: continue
        if dcm_path: break

    T = extract_affine_matrix(dcm_path)
    spacing = tuple(np.linalg.norm(T[i, :3]) for i in range(3))
    
    # 2. Generación Malla 3D en Alta Resolución (step_size=1)
    print("-> Generando Malla 3D en Alta Resolución (Marching Cubes)...")
    verts, faces, normals, _ = marching_cubes(
        volume=binary_mask, level=0.5, spacing=spacing, step_size=1
    )
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # 4. Limpieza de Islas Inteligente (Top N componentes)
    print("-> Limpiando ruido (conservando los 5 componentes anatómicos más grandes)...")
    components = mesh.split(only_watertight=False)
    if len(components) > 5:
        # Ordenar componentes por área usando sorted() ya que components es un array de numpy
        sorted_components = sorted(components, key=lambda c: c.area, reverse=True)
        # Quedarnos solo con el Top 5 (Pelvis, Sacro, Fémures, etc.)
        mesh = trimesh.util.concatenate(sorted_components[:5])
    elif len(components) > 1:
        mesh = trimesh.util.concatenate(components)

    # 5. Optimización e Isotropía
    print("-> Optimizando isotropía (Voronoi 20k)...")
    mesh = optimize_mesh_quality(mesh, target_n_vertices=20000)
    
    # 6. Suavizado de Taubin (No encoge)
    print("-> Aplicando Suavizado de Taubin...")
    trimesh.smoothing.filter_taubin(mesh, iterations=15)
    
    final_file = os.path.join(out_dir, "hueso_completo_FEM.stl")
    mesh.export(final_file)
    
    print(f"   [✓] ÉXITO: {final_file}")