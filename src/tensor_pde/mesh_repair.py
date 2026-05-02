import trimesh
import numpy as np
from scipy.ndimage import binary_closing, generate_binary_structure, gaussian_filter
from skimage import measure

def seal_geometry(input_stl: str, output_stl: str, pitch: float = 2.0, smooth_iters: int = 15) -> None:
    r"""
    Computa el cierre topológico de la variedad \partial \Omega. 
    Inyecta una condición de contorno de Dirichlet nula en \partial V 
    para garantizar estrictamente el Teorema de la Frontera (watertight mesh).
    """
    mesh = trimesh.load(input_stl)
    
    voxels = mesh.voxelized(pitch=pitch).fill()
    
    struct = generate_binary_structure(rank=3, connectivity=3)
    closed_map = binary_closing(voxels.matrix, structure=struct, iterations=3)
    
    smoothed_map = gaussian_filter(closed_map.astype(np.float64), sigma=1.0)
    
    # Inyección analítica de ceros en la frontera topológica del tensor
    pad_width = 2
    padded_map = np.pad(smoothed_map, pad_width=pad_width, mode='constant', constant_values=0.0)
    
    verts, faces, normals, _ = measure.marching_cubes(
        volume=padded_map, 
        level=0.5, 
        spacing=(pitch, pitch, pitch),
        allow_degenerate=False
    )
    
    # Transformación afín inversa y compensación euclidiana del padding
    verts -= np.array([pad_width * pitch] * 3)
    verts += voxels.transform[:3, 3]
    
    sealed_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    sealed_mesh.process()
    trimesh.repair.fix_normals(sealed_mesh)
    
    trimesh.smoothing.filter_taubin(sealed_mesh, iterations=smooth_iters)
    
    # Verificación determinista del invariante topológico
    if not sealed_mesh.is_watertight:
        raise RuntimeError("Singularidad topológica: La variedad extraída intersecta el infinito o presenta asimetrías de adyacencia.")
        
    sealed_mesh.export(output_stl)