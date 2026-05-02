import SimpleITK as sitk
import numpy as np
import trimesh
from skimage import measure, filters
from scipy.ndimage import gaussian_filter

def dicom_to_stl_otsu(dicom_dir: str, output_stl: str, sigma: float = 1.0) -> None:
    r"""
    Operador automatizado de extracción \mathbf{X} \to \partial \Omega.
    Evalúa la varianza de Otsu y componentes conexos para evadir la inyección de escalares manuales.
    """
    print("Ensamblando tensor métrico...")
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
    
    if not series_IDs:
        raise RuntimeError(r"Singularidad: Conjunto \emptyset de identificadores DICOM.")
        
    # Aislamiento del subespacio con máxima cardinalidad en z
    target_series_id = max(series_IDs, key=lambda s: len(reader.GetGDCMSeriesFileNames(dicom_dir, s)))
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, target_series_id)
    reader.SetFileNames(dicom_names)
    image_tensor = reader.Execute()
    
    # El tensor en SimpleITK se proyecta en \mathbb{R}^{N_z \times N_y \times N_x}
    voxel_data = sitk.GetArrayFromImage(image_tensor).astype(np.float64)
    pitch_x, pitch_y, pitch_z = image_tensor.GetSpacing()
    
    print("Evaluando partición óptima por Varianza de Otsu...")
    tau_opt = filters.threshold_otsu(voxel_data)
    print(rf"Umbral óptimo inferido: \tau^* = {tau_opt:.2f}")
    
    # Mapeo al subespacio booleano
    binary_tensor = (voxel_data > tau_opt).astype(np.uint8)
    
    print("Aislando variedad principal (Componentes Conexos)...")
    labels = measure.label(binary_tensor, connectivity=3)
    props = measure.regionprops(labels)
    
    if not props:
        raise RuntimeError(r"Singularidad Topológica: Operador nulo tras binarización.")
        
    # Extracción de la subvariedad con volumen máximo \max(|C_k|)
    largest_label = max(props, key=lambda p: p.area).label
    main_component = (labels == largest_label).astype(np.float64)
    
    print(rf"Inyectando operador G_\sigma con \sigma = {sigma}...")
    smoothed_map = gaussian_filter(main_component, sigma=sigma)
    
    pad_width = 3
    padded_map = np.pad(smoothed_map, pad_width=pad_width, mode='constant', constant_values=0.0)
    
    print("Computando complejo simplicial (Marching Cubes)...")
    verts, faces, normals, _ = measure.marching_cubes(
        volume=padded_map, 
        level=0.5, 
        spacing=(pitch_z, pitch_y, pitch_x), 
        allow_degenerate=False
    )
    
    # Corrección de traslación afín
    verts -= np.array([pad_width * pitch_z, pad_width * pitch_y, pad_width * pitch_x])
    
    # Biyección de coordenadas (Z, Y, X) \to (X, Y, Z) para el homeomorfismo euclidiano
    verts_xyz = verts[:, [2, 1, 0]]
    normals_xyz = normals[:, [2, 1, 0]]
    
    sealed_mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, vertex_normals=normals_xyz)
    sealed_mesh.process()
    trimesh.repair.fix_normals(sealed_mesh)
    
    if not sealed_mesh.is_watertight:
        print(r"Advertencia: Homeomorfismo imperfecto. \partial \Omega posee discontinuidades.")
        
    sealed_mesh.export(output_stl)
    print(rf"Variedad \partial \Omega exportada exitosamente a: {output_stl}")

if __name__ == "__main__":
    # Inyectar el vector de ruta absoluta del TAC del modelo de pelvis
    dir_dicom = "data/01_raw/paciente1"
    out_stl = "pelvis_reconstruido.stl"
    
    dicom_to_stl_otsu(dir_dicom, out_stl)