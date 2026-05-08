import numpy as np
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
import trimesh
import os
import pydicom
from src.tensor_pde.mesh_repair import optimize_mesh_quality, seal_geometry
from src.tensor_pde.io_module import extract_affine_matrix, assemble_tensor_and_hu

def process_and_save_dl_mesh(prob_volume: np.ndarray, dicom_dir: str, out_dir: str) -> None:
    r"""
    Versión 'Cerebro Otsu': Sincronizada con el éxito de test_inference.py
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Filtro Físico (HU > 200) - Recuperamos datos crudos
    print("-> Aplicando filtro de densidad física (HU)...")
    X_orig = assemble_tensor_and_hu(dicom_dir)
    # Sincronizamos dimensiones por si hubo algún padding
    if X_orig.shape != prob_volume.shape:
        # Si hay diferencia, tomamos el centro o ajustamos
        from scipy.ndimage import zoom
        factors = [p/o for p, o in zip(prob_volume.shape, X_orig.shape)]
        X_orig = zoom(X_orig, factors, order=1)
        
    prob_volume[X_orig < 200] = 0
    
    # 2. Umbral Inteligente (Otsu)
    print("-> Calculando umbral óptimo (Otsu)...")
    prob_samples = prob_volume[prob_volume > 0.05]
    if len(prob_samples) > 1000:
        tau = threshold_otsu(prob_samples)
        tau = max(0.3, min(tau, 0.8)) # Límite de seguridad
    else:
        tau = 0.5
    print(f"   [+] Umbral calculado: {tau:.3f}")
    
    # 3. Extracción de Malla
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
    
    print("-> Generando Malla 3D (Marching Cubes)...")
    verts, faces, normals, _ = marching_cubes(
        volume=binary_mask, level=0.5, spacing=spacing, step_size=2
    )
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # 4. Limpieza de Islas
    print("-> Limpiando fragmentos huérfanos...")
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        max_area = max(c.area for c in components)
        mesh = trimesh.util.concatenate([c for c in components if c.area > max_area * 0.1])

    # 5. Optimización e Isotropía
    print("-> Optimizando isotropía (Voronoi 20k)...")
    mesh = optimize_mesh_quality(mesh, target_n_vertices=20000)
    
    # 6. Suavizado de Taubin (No encoge)
    print("-> Aplicando Suavizado de Taubin...")
    trimesh.smoothing.filter_taubin(mesh, iterations=15)
    
    final_file = os.path.join(out_dir, "hueso_completo_FEM.stl")
    mesh.export(final_file)
    
    print(f"   [✓] ÉXITO: {final_file}")