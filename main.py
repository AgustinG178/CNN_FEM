import os
import numpy as np
import trimesh
from src.isolate_main import extract_anatomical_domains, optimize_mesh_quality
from src.tensor_pde.material_mapping import generate_comsol_material_field
from src.tensor_pde.comsol_mapper import map_all_selections, export_heterogeneous_field
from src.tensor_pde.io_module import assemble_tensor_and_hu, extract_affine_matrix
from src.neural_manifold.inference import predict_volume_from_dicom
from src.neural_manifold.segment_pde import process_and_save_dl_mesh

def export_dicom_to_nifti(dicom_dir: str, output_path: str) -> str:
    r"""
    Ensambla el tensor volumétrico HU desde DICOM y lo persiste como NIfTI (.nii.gz),
    generando la matriz afín necesaria para el mapeo de propiedades biomecánicas.
    """
    import nibabel as nib
    
    X_hu = assemble_tensor_and_hu(dicom_dir)
    
    # Extraer la matriz afín del primer DICOM
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.lower().endswith('.dcm')]
    if not dicom_files:
        raise ValueError("No se encontraron DICOMs para extraer la matriz afín.")
    
    T = extract_affine_matrix(dicom_files[0])
    
    # NIfTI espera la convención RAS, pero nuestra matriz T ya está en LPS.
    # Para compatibilidad con nibabel, invertimos los dos primeros ejes (L->R, P->A).
    T_ras = T.copy()
    T_ras[0, :] *= -1  # L -> R
    T_ras[1, :] *= -1  # P -> A
    
    nifti_img = nib.Nifti1Image(X_hu, affine=T_ras)
    nib.save(nifti_img, output_path)
    print(f"-> Volumen NIfTI exportado: {output_path} (shape: {X_hu.shape})")
    
    return output_path

def main():
    r"""
    Orquesta la extracción de la variedad topológica \partial \Omega, aplica remallado 
    isotrópico para estabilidad del Jacobiano en FEM, y proyecta el campo de rigidez 
    heterogéneo E(HU) junto con las selecciones de frontera \Gamma.
    """

    r"""
    Resolución dinámica del vector espacial raíz y ensamblado de rutas.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # MODO DE EJECUCIÓN: "CLASSIC" o "DEEP_LEARNING"
    PIPELINE_MODE = "DEEP_LEARNING"
    
    dicom_dir = os.path.join(BASE_DIR, "data", "01_raw", "Fantoma_Pelvis")
    out_dir = os.path.join(BASE_DIR, "data", "02_processed")
    dominios_dir = os.path.join(out_dir, "dominios_anatomicos")
    selections_dir = os.path.join(out_dir, "selections_comsol")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(selections_dir, exist_ok=True)

    try:
        print(f"--- INICIANDO PIPELINE DE BIOMECÁNICA COMPUTACIONAL ({PIPELINE_MODE}) ---")
        
        if PIPELINE_MODE == "CLASSIC":
            print("1. Segmentando dominios anatómicos y optimizando mallas (PyACVD/TotalSegmentator)...")
            extract_anatomical_domains(dicom_dir, dominios_dir, sigma=1.75)
        
        elif PIPELINE_MODE == "DEEP_LEARNING":
            print("1. Ejecutando Inferencia Topológica mediante CNN (UNet3D)...")
            model_path = os.path.join(BASE_DIR, "data", "03_models", "unet_bone_topology.pth")
            
            # 1.a Inferencia de volumen a partir de DICOM
            binary_mask = predict_volume_from_dicom(
                dicom_dir=dicom_dir, 
                model_path=model_path,
                patch_size=(64, 64, 64)
            )
            
            # 1.b Generación de STL separados (Pelvis, Fémur L, Fémur R) + Watertight
            process_and_save_dl_mesh(
                binary_mask=binary_mask, 
                dicom_dir=dicom_dir, 
                out_dir=dominios_dir
            )

        # 2. Exportar volumen HU como NIfTI para el mapeo de materiales
        print("2. Exportando volumen DICOM como NIfTI para mapeo de rigidez...")
        nifti_volume = os.path.join(out_dir, "ct_volume.nii.gz")
        export_dicom_to_nifti(dicom_dir, nifti_volume)
        
        print("3. Generando campo escalar de rigidez heterogénea E(HU)...")
        material_output = os.path.join(out_dir, "mapa_elasticidad_heterogeneo.txt")
        generate_comsol_material_field(nifti_volume, material_output)

        print("4. Identificando subvariedades de frontera para condiciones de contorno...")
        map_all_selections(dominios_dir, selections_dir)

        print(f"--- PROCESAMIENTO FINALIZADO ---")
        print(f"Archivos listos para importación en COMSOL en: {out_dir}")

    except Exception as e:
        print(f"Divergencia analítica durante la orquestación: {e}")

if __name__ == "__main__":
    main()