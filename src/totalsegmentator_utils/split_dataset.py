"""
split_dataset.py
----------------
Divide el dataset procesado de TotalSegmentator en 3 particiones:
  - Train:      80% (~982 pacientes) - Lo que ve la red durante el entrenamiento
  - Validation: 10% (~123 pacientes) - Para medir Dice real durante el entrenamiento
  - Test:       10% (~123 pacientes) - NUNCA se toca hasta el paper final

Genera un archivo JSON con los splits para garantizar reproducibilidad.
El mismo split debe usarse en V4, V5 y V6 para que las métricas sean comparables.
"""

import os
import json
import random
import glob
import nibabel as nib
import numpy as np
from datetime import datetime

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
PROCESSED_DIR  = "data/05_totalsegmentator/processed"
SPLIT_OUT_PATH = "data/05_totalsegmentator/dataset_split.json"
RANDOM_SEED    = 42   # Semilla fija para reproducibilidad total

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10    # Debe sumar 1.0


def compute_bone_ratio(mask_path: str) -> float:
    """
    Calcula la proporción de vóxeles de hueso en la máscara.
    Sirve para hacer un split ESTRATIFICADO: que Train/Val/Test tengan
    la misma distribución de pacientes 'con mucho hueso' y 'con poco hueso'.
    Pacientes con fracturas o anomalías suelen tener máscaras más pequeñas.
    """
    img = nib.load(mask_path)
    data = img.get_fdata()
    total_voxels = data.size
    bone_voxels = np.sum(data > 0)
    return bone_voxels / total_voxels


def stratified_split(subjects: list, train_r: float, val_r: float, seed: int):
    """
    Divide la lista de sujetos de forma ESTRATIFICADA por ratio óseo.
    Esto garantiza que los 3 splits tengan la misma distribución anatómica.
    """
    random.seed(seed)
    
    # Ordenamos por ratio óseo y dividimos en 10 bins (deciles)
    subjects_sorted = sorted(subjects, key=lambda x: x['bone_ratio'])
    n = len(subjects_sorted)
    n_bins = 10
    bin_size = n // n_bins
    
    train, val, test = [], [], []
    
    for i in range(n_bins):
        bin_subjects = subjects_sorted[i * bin_size : (i + 1) * bin_size]
        if i == n_bins - 1:
            # El último bin absorbe los sujetos restantes
            bin_subjects = subjects_sorted[i * bin_size:]
        
        random.shuffle(bin_subjects)
        
        n_bin = len(bin_subjects)
        n_train = int(n_bin * train_r)
        n_val   = int(n_bin * val_r)
        
        train += bin_subjects[:n_train]
        val   += bin_subjects[n_train:n_train + n_val]
        test  += bin_subjects[n_train + n_val:]
    
    return train, val, test


def main():
    print("=" * 60)
    print(" GENERANDO SPLIT ESTRATIFICADO DEL DATASET V4")
    print("=" * 60)
    
    # 1. Listar todos los pacientes procesados
    patient_dirs = sorted(glob.glob(os.path.join(PROCESSED_DIR, "s*")))
    
    if not patient_dirs:
        raise FileNotFoundError(
            f"No se encontraron pacientes en '{PROCESSED_DIR}'.\n"
            f"Ejecutá primero extract_bones.py."
        )
    
    print(f"[*] Pacientes encontrados: {len(patient_dirs)}")
    
    # 2. Calcular ratio óseo para estratificación
    print("[*] Calculando ratio óseo para estratificación (puede tardar unos minutos)...")
    subjects = []
    errors = []
    
    for p_dir in patient_dirs:
        patient_id = os.path.basename(p_dir)
        ct_path   = os.path.join(p_dir, "ct.nii.gz")
        mask_path = os.path.join(p_dir, "bone_mask.nii.gz")
        
        if not os.path.exists(ct_path) or not os.path.exists(mask_path):
            errors.append(patient_id)
            continue
        
        try:
            bone_ratio = compute_bone_ratio(mask_path)
            subjects.append({
                "id":          patient_id,
                "ct_path":     ct_path,
                "mask_path":   mask_path,
                "bone_ratio":  round(float(bone_ratio), 6)
            })
        except Exception as e:
            errors.append(patient_id)
            print(f"  [!] Error en {patient_id}: {e}")
    
    print(f"[✓] Pacientes válidos: {len(subjects)} | Errores: {len(errors)}")
    if errors:
        print(f"  [!] Pacientes con errores (excluidos): {errors}")
    
    # 3. Split estratificado
    train, val, test = stratified_split(subjects, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    
    print(f"\n[*] División del Dataset:")
    print(f"  Train:      {len(train):>4} pacientes  ({len(train)/len(subjects)*100:.1f}%)")
    print(f"  Validation: {len(val):>4} pacientes  ({len(val)/len(subjects)*100:.1f}%)")
    print(f"  Test:       {len(test):>4} pacientes  ({len(test)/len(subjects)*100:.1f}%)")
    
    # 4. Guardar como JSON reproducible
    split_data = {
        "metadata": {
            "created_at":      datetime.now().isoformat(),
            "random_seed":     RANDOM_SEED,
            "total_patients":  len(subjects),
            "train_ratio":     TRAIN_RATIO,
            "val_ratio":       VAL_RATIO,
            "test_ratio":      TEST_RATIO,
            "description":     (
                "Split estratificado por ratio óseo. "
                "Usar este mismo archivo para V4, V5 y V6 "
                "para garantizar comparabilidad de métricas."
            )
        },
        "train":      train,
        "validation": val,
        "test":       test,
        "errors":     errors
    }
    
    os.makedirs(os.path.dirname(SPLIT_OUT_PATH), exist_ok=True)
    with open(SPLIT_OUT_PATH, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"\n[✓] Split guardado en: {SPLIT_OUT_PATH}")
    print(f"\n[!] IMPORTANTE: No modificar ni regenerar este archivo.")
    print(f"    Todas las versiones (V4, V5, V6) deben usar el mismo split")
    print(f"    para que las métricas Dice sean científicamente comparables.")


if __name__ == "__main__":
    main()
