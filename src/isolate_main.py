import os
import tempfile
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import trimesh
from skimage import measure
from scipy.ndimage import gaussian_filter
from totalsegmentator.python_api import totalsegmentator
import pyacvd
import pyvista as pv

def optimize_mesh_quality(mesh_trimesh, target_n_vertices=15000):
    r"""
    Aplica una partición de Voronoi sobre la variedad \partial \Omega 
    para garantizar isotropía en los tensores de deformación locales.
    """
    pv_mesh = pv.wrap(mesh_trimesh)
    clus = pyacvd.Clustering(pv_mesh)
    clus.subdivide(3) 
    clus.cluster(target_n_vertices)
    optimized_pv_mesh = clus.create_mesh()
    
    return trimesh.Trimesh(vertices=optimized_pv_mesh.points, 
                           faces=optimized_pv_mesh.faces.reshape(-1, 4)[:, 1:])

def extract_anatomical_domains(dicom_dir: str, output_dir: str, sigma: float = 1.75) -> None:
    r"""
    Genera los complejos simpliciales y preserva el tensor NIfTI 
    en el directorio principal para el mapeo elastoplástico.
    """
    ANATOMY_LABELS = ["femur_right", "femur_left", "hip_right", "hip_left", "sacrum", "vertebrae_L5"]
    os.makedirs(output_dir, exist_ok=True)
    
    r"""
    Proyección escalar fuera del dominio efímero.
    """
    nifti_persistent_path = os.path.join(os.path.dirname(output_dir), "ct_volume.nii.gz")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        seg_output_dir = os.path.join(tmp_dir, "segmentations") 
        os.makedirs(seg_output_dir, exist_ok=True)
        
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
        target_series_id = max(series_IDs, key=lambda s: len(reader.GetGDCMSeriesFileNames(dicom_dir, s)))
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_dir, target_series_id))
        
        sitk.WriteImage(reader.Execute(), nifti_persistent_path)
        
        totalsegmentator(nifti_persistent_path, seg_output_dir, fast=False, ml=False, roi_subset=ANATOMY_LABELS)
        
        for anatomy_name in ANATOMY_LABELS:
            mask_path = os.path.join(seg_output_dir, f"{anatomy_name}.nii.gz")
            if not os.path.exists(mask_path): continue
                
            img = nib.load(mask_path)
            domain_mask = (img.get_fdata() > 0).astype(np.float64)
            if np.max(domain_mask) == 0.0: continue
                
            affine = img.affine
            pitch_x, pitch_y, pitch_z = [np.linalg.norm(affine[i, :3]) for i in range(3)]
            
            smoothed_map = gaussian_filter(domain_mask, sigma=sigma)
            pad_width = 3
            padded_map = np.pad(smoothed_map, pad_width=pad_width, mode='constant', constant_values=0.0)
            
            max_val = np.max(padded_map)
            min_val = np.min(padded_map)
            
            if max_val <= 0.5 or max_val == min_val:
                print(rf"   [!] Singularidad: {anatomy_name} con señal insuficiente tras G_\sigma. Omitiendo...")
                continue
            
            verts, faces, normals, _ = measure.marching_cubes(
                volume=padded_map, level=0.5, spacing=(pitch_x, pitch_y, pitch_z), allow_degenerate=False
            )
            
            verts -= np.array([pad_width * pitch_x, pad_width * pitch_y, pad_width * pitch_z])
            coords = np.c_[verts, np.ones(len(verts))]
            global_verts = (affine @ coords.T).T[:, :3]
            
            mesh = trimesh.Trimesh(vertices=global_verts, faces=faces)
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
            mesh.merge_vertices() 
            
            print(f"-> Optimizando calidad de malla para: {anatomy_name}")
            mesh_opt = optimize_mesh_quality(mesh, target_n_vertices=15000)
            
            out_file = os.path.join(output_dir, f"dominio_{anatomy_name}.stl")
            mesh_opt.export(out_file)