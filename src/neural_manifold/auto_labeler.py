import os
import tempfile
import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator
from src.tensor_pde.io_module import assemble_tensor_and_hu, extract_affine_matrix

def generate_ground_truth_for_all_patients(raw_dir: str, out_dir: str):
    r"""
    Procesa masivamente el directorio de pacientes, autogenerando las 
    etiquetas de entrenamiento mediante TotalSegmentator y fusionándolas 
    en un único tensor geométrico.
    """
    ANATOMY_LABELS = ["femur_right", "femur_left", "hip_right", "hip_left", "sacrum", "vertebrae_L5"]
    os.makedirs(out_dir, exist_ok=True)
    
    patient_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    print(f"--- INICIANDO AUTO-ETIQUETADO PARA {len(patient_dirs)} PACIENTES ---")
    
    for pid in patient_dirs:
        dicom_path = os.path.join(raw_dir, pid)
        out_mask_path = os.path.join(out_dir, f"{pid}_mask.npy")
        
        if os.path.exists(out_mask_path):
            print(f"-> {pid}: Máscara ya existe. Omitiendo...")
            continue
            
        print(f"\n-> Procesando paciente: {pid}")
        
        dicom_files = []
        for root, dirs, files in os.walk(dicom_path):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, f))
        
        if not dicom_files:
            print(f"   [!] No hay DICOMs en {pid}. Omitiendo.")
            continue
            
        try:
            # 1. Ensamblado con nuestra función para garantizar orientación X,Y,Z
            X_tensor = assemble_tensor_and_hu(dicom_path)
            
            # Buscar una matriz afín válida tolerando topogramas sin orientación
            T = None
            for dcm_file in dicom_files:
                try:
                    T = extract_affine_matrix(dcm_file)
                    break
                except Exception:
                    continue
            
            if T is None:
                raise ValueError("No se encontró ningún DICOM con ImageOrientationPatient válido.")
            
            # 2. Creación del NIfTI temporal
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_nifti = os.path.join(tmp_dir, "temp_ct.nii.gz")
                seg_output = os.path.join(tmp_dir, "segmentations")
                os.makedirs(seg_output, exist_ok=True)
                
                nifti_img = nib.Nifti1Image(X_tensor, T)
                nib.save(nifti_img, tmp_nifti)
                
                # 3. Inferencia
                print("   Ejecutando IA preentrenada (TotalSegmentator)...")
                totalsegmentator(tmp_nifti, seg_output, fast=False, ml=False, roi_subset=ANATOMY_LABELS)
                
                # 4. Fusión de máscaras
                Y_tensor = np.zeros_like(X_tensor, dtype=np.float32)
                
                # Definimos el NIfTI de referencia (el que creamos nosotros)
                ref_img = nifti_img
                
                found_any = False
                for anatomy in ANATOMY_LABELS:
                    mask_file = os.path.join(seg_output, f"{anatomy}.nii.gz")
                    if os.path.exists(mask_file):
                        mask_img = nib.load(mask_file)
                        
                        # ALINEACIÓN GARANTIZADA: Re-muestrear la máscara al espacio del CT original
                        from nibabel.processing import resample_from_to
                        resampled_mask_img = resample_from_to(mask_img, ref_img, order=0) # order=0 para etiquetas (nearest)
                        anatomy_mask = resampled_mask_img.get_fdata()
                        
                        # Suma booleana (OR lógico)
                        Y_tensor += (anatomy_mask > 0).astype(np.float32)
                        found_any = True
                
                if found_any:
                    Y_tensor = (Y_tensor > 0).astype(np.float32) # Binarizar estrictamente a 1 o 0
                    np.save(out_mask_path, Y_tensor)
                    print(f"   [OK] Máscara autogenerada y guardada en {out_mask_path}")
                else:
                    print(f"   [!] TotalSegmentator no encontró huesos en {pid}. Guardando máscara vacía.")
                    np.save(out_mask_path, Y_tensor)

        except Exception as e:
            print(f"   [ERROR] Falló el procesamiento de {pid}: {e}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DIR = os.path.join(BASE_DIR, "data", "01_raw")
    GT_DIR = os.path.join(BASE_DIR, "data", "01_ground_truth")
    
    generate_ground_truth_for_all_patients(RAW_DIR, GT_DIR)
