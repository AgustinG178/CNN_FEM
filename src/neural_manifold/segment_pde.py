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
    # Ordenar por cantidad de caras (descendente)
    components.sort(key=lambda c: len(c.faces), reverse=True)
    
    # Retornar las 3 componentes más grandes (esperamos Pelvis, Fémur 1, Fémur 2)
    # Si por algún motivo hay menos de 3, retornamos las que haya.
    top_components = components[:3]
    return top_components

import os
import nibabel as nib
from src.tensor_pde.mesh_repair import optimize_mesh_quality, seal_geometry
from src.tensor_pde.io_module import extract_affine_matrix

def process_and_save_dl_mesh(binary_mask: np.ndarray, dicom_dir: str, out_dir: str) -> None:
    r"""
    Toma la predicción binaria, extrae la métrica del espacio DICOM, computa la malla STL,
    separa los dominios anatómicos (Pelvis, Fémures), optimiza su isotropía y asegura
    el cierre topológico (Watertight) para su uso en COMSOL.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Búsqueda recursiva de DICOMs válidos (filtrando localizers/scouts)
    import pydicom
    valid_dcm = None
    for root, dirs, files in os.walk(dicom_dir):
        for f in files:
            if f.lower().endswith('.dcm'):
                fpath = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                    image_type = getattr(ds, 'ImageType', [])
                    if 'LOCALIZER' not in image_type and hasattr(ds, 'ImageOrientationPatient'):
                        valid_dcm = fpath
                        break
                except Exception:
                    continue
        if valid_dcm:
            break
    
    if not valid_dcm:
        raise ValueError("No se encontraron DICOMs válidos (con ImageOrientationPatient) para extraer la matriz afín.")
        
    T = extract_affine_matrix(valid_dcm)
    pitch_x, pitch_y, pitch_z = [np.linalg.norm(T[i, :3]) for i in range(3)]
    
    print("-> Extrayendo variedad de frontera (Marching Cubes)...")
    components = extract_boundary_manifold(binary_mask, tau=0.5, spacing=(pitch_x, pitch_y, pitch_z))
    
    if len(components) < 1:
        raise RuntimeError("No se detectó ninguna topología ósea en la predicción.")
        
    # Clasificación heurística espacial:
    # 1. El más grande es la Pelvis (Sacrum + Ilíacos)
    pelvis_mesh = components[0]
    labeled_meshes = [("sacrum", pelvis_mesh)]
    
    # 2. Si hay más componentes, son los fémures. Los clasificamos por su centroide en X.
    if len(components) == 3:
        femur_a = components[1]
        femur_b = components[2]
        
        # En coordenadas físicas, X suele definir Izquierda/Derecha.
        if femur_a.centroid[0] < femur_b.centroid[0]:
            labeled_meshes.append(("hip_right", femur_a))
            labeled_meshes.append(("hip_left", femur_b))
        else:
            labeled_meshes.append(("hip_left", femur_a))
            labeled_meshes.append(("hip_right", femur_b))
    else:
        print(f"[!] Advertencia: Se detectaron {len(components)} componentes óseos en lugar de 3.")
        for i in range(1, len(components)):
            labeled_meshes.append((f"bone_{i}", components[i]))

    for name, raw_mesh in labeled_meshes:
        print(f"-> Optimizando isotropía de la malla '{name}' (PyACVD)...")
        # Target vertices puede ser un parámetro, usamos 15000 por defecto de isolate_main
        optimized_mesh = optimize_mesh_quality(raw_mesh, target_n_vertices=15000)
        
        # Guardado temporal de la malla abierta
        temp_file = os.path.join(out_dir, f"dominio_{name}_raw.stl")
        final_file = os.path.join(out_dir, f"dominio_{name}.stl")
        
        optimized_mesh.export(temp_file)
        
        print(f"-> Sellando geometría (Watertight) para COMSOL en '{name}'...")
        try:
            seal_geometry(temp_file, final_file, pitch=2.0, smooth_iters=10)
            os.remove(temp_file) # Limpieza
            print(f"   [+] Malla sellada exportada en: {final_file}")
        except Exception as e:
            print(f"   [-] Falló el sellado de '{name}': {e}. Usando versión sin sellar.")
            os.rename(temp_file, final_file)