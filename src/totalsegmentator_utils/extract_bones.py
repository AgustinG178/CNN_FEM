import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

def process_ts_patient(patient_dir, output_dir):
    """
    Extrae sacrum, hips y femurs de la estructura de TotalSegmentator
    y los fusiona en una máscara binaria.
    """
    patient_id = os.path.basename(patient_dir)
    seg_dir = os.path.join(patient_dir, 'segmentations')
    
    # Archivos que componen nuestro sistema óseo de interés
    TARGET_MASKS = [
        'sacrum.nii.gz',
        'hip_left.nii.gz',
        'hip_right.nii.gz',
        'femur_left.nii.gz',
        'femur_right.nii.gz'
    ]
    
    # 1. Verificar si existe la imagen de CT original
    ct_path = os.path.join(patient_dir, 'ct.nii.gz')
    if not os.path.exists(ct_path):
        return False
    
    combined_mask = None
    affine = None
    header = None
    
    # 2. Fusionar las máscaras
    found_any = False
    for mask_name in TARGET_MASKS:
        mask_path = os.path.join(seg_dir, mask_name)
        if os.path.exists(mask_path):
            img = nib.load(mask_path)
            data = img.get_fdata() > 0
            
            if combined_mask is None:
                combined_mask = np.zeros_like(data, dtype=np.uint8)
                affine = img.affine
                header = img.header
            
            combined_mask[data] = 1
            found_any = True
            
    if not found_any:
        return False
    
    # 3. Guardar resultados en la carpeta procesada
    pat_out_dir = os.path.join(output_dir, patient_id)
    os.makedirs(pat_out_dir, exist_ok=True)
    
    # Guardar Máscara Fusionada
    mask_out_img = nib.Nifti1Image(combined_mask, affine, header)
    nib.save(mask_out_img, os.path.join(pat_out_dir, "bone_mask.nii.gz"))
    
    # Crear un enlace simbólico o copiar el CT original para tenerlo a mano
    # Usamos copia para evitar problemas de rutas relativas en el entrenamiento
    import shutil
    shutil.copy(ct_path, os.path.join(pat_out_dir, "ct.nii.gz"))
    
    return True

def main():
    RAW_DIR = "data/05_totalsegmentator/raw"
    PROCESSED_DIR = "data/05_totalsegmentator/processed"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Buscamos todas las carpetas de sujetos (que empiezan con 's')
    subjects = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d)) and d.startswith('s')]
    
    print(f"[*] Iniciando destilación de {len(subjects)} pacientes...")
    
    success_count = 0
    for s in tqdm(subjects):
        try:
            if process_ts_patient(os.path.join(RAW_DIR, s), PROCESSED_DIR):
                success_count += 1
        except Exception as e:
            print(f"\n[!] Error en paciente {s}: {e}")
            
    print(f"\n[✓] Proceso finalizado. Pacientes destilados con éxito: {success_count}/{len(subjects)}")

if __name__ == "__main__":
    main()
