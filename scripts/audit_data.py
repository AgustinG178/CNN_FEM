import numpy as np
import glob
import os

PATCH_DIR = "data/04_training_patches/masks"

def audit_masks():
    mask_paths = glob.glob(os.path.join(PATCH_DIR, "*.npy"))
    total = len(mask_paths)
    empty_count = 0
    
    print(f"-> Auditando {total} máscaras...")
    
    # Revisamos los primeros 500 para tener una estadística
    for i in range(min(500, total)):
        mask = np.load(mask_paths[i])
        if np.sum(mask) == 0:
            empty_count += 1
            
    percentage = (empty_count / min(500, total)) * 100
    print(f"-> Resultado: {empty_count} de {min(500, total)} están VACÍAS ({percentage:.1f}%)")
    
    if percentage > 90:
        print("[!] ALERTA CRÍTICA: Casi todas las máscaras están vacías. El entrenamiento no sirve.")
    else:
        print("[✓] Las máscaras parecen tener datos. El problema puede ser el desfasaje de nombres.")

if __name__ == "__main__":
    audit_masks()
