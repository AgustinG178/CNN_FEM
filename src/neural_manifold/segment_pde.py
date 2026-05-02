import numpy as np
from skimage.measure import marching_cubes
import trimesh

def extract_boundary_manifold(X_tensor: np.ndarray, tau: float, spacing: tuple) -> trimesh.Trimesh:
    r"""
    Computa la variedad de frontera parcial \partial \Omega y aplica un filtro de 
    Teoría de Grafos para aislar la componente conexa de máxima cardinalidad,
    purgando subespacios no homeomorfos al dominio de interés.
    """
    verts, faces, normals, values = marching_cubes(
        volume=X_tensor, 
        level=tau, 
        spacing=spacing,
        step_size=1,
        allow_degenerate=False
    )
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    if mesh.volume < 0:
        mesh.invert()
        
    components = mesh.split(only_watertight=False)
    main_component = max(components, key=lambda c: len(c.faces))
    return main_component

import os
import nibabel as nib
from src.isolate_main import optimize_mesh_quality
from src.tensor_pde.io_module import extract_affine_matrix

def process_and_save_dl_mesh(binary_mask: np.ndarray, dicom_dir: str, out_dir: str) -> None:
    r"""
    Toma la predicción binaria, extrae la métrica del espacio DICOM, computa la malla STL,
    optimiza su isotropía y la persiste para uso en COMSOL.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    if not dicom_files:
        raise ValueError("No se encontraron DICOMs para extraer matriz afín.")
        
    T = extract_affine_matrix(dicom_files[0])
    pitch_x, pitch_y, pitch_z = [np.linalg.norm(T[i, :3]) for i in range(3)]
    
    print("-> Extrayendo variedad de frontera (Marching Cubes)...")
    raw_mesh = extract_boundary_manifold(binary_mask, tau=0.5, spacing=(pitch_x, pitch_y, pitch_z))
    
    print("-> Optimizando isotropía de la malla (PyACVD)...")
    # Target vertices puede ser un parámetro, usamos 15000 por defecto de isolate_main
    optimized_mesh = optimize_mesh_quality(raw_mesh, target_n_vertices=15000)
    
    out_file = os.path.join(out_dir, "dominio_dl_reconstructed.stl")
    optimized_mesh.export(out_file)
    print(f"-> Malla reconstruida exportada exitosamente en: {out_file}")