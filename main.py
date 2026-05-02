import os
import numpy as np
import trimesh
from src.isolate_main import extract_anatomical_domains, optimize_mesh_quality
from src.tensor_pde.material_mapping import generate_comsol_material_field
from src.tensor_pde.comsol_mapper import map_all_selections
from src.neural_manifold.inference import predict_volume_from_dicom
from src.neural_manifold.segment_pde import process_and_save_dl_mesh

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
            
            # 1.b Generación de STL optimizado
            process_and_save_dl_mesh(
                binary_mask=binary_mask, 
                dicom_dir=dicom_dir, 
                out_dir=dominios_dir
            )

        print("2. Generando campo escalar de rigidez heterogénea E(HU)...")
        # En modo DL el NIfTI se puede exportar en el futuro, o depender del original
        nifti_volume = os.path.join(out_dir, "ct_volume.nii.gz")
        material_output = os.path.join(out_dir, "mapa_elasticidad_heterogeneo.txt")
        
        if os.path.exists(nifti_volume):
            generate_comsol_material_field(nifti_volume, material_output)
        else:
            print("[!] Advertencia: No se encontró ct_volume.nii.gz. (Nota: en modo DL puro puede ser necesario exportarlo).")

        print("3. Identificando subvariedades de frontera para condiciones de contorno...")
        map_all_selections(dominios_dir, selections_dir)

        print(f"--- PROCESAMIENTO FINALIZADO ---")
        print(f"Archivos listos para importación en COMSOL en: {out_dir}")

    except Exception as e:
        print(f"Divergencia analítica durante la orquestación: {e}")

if __name__ == "__main__":
    main()