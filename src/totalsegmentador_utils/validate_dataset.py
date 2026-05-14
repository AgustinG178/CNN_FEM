"""
validate_dataset.py
-------------------
Valida la integridad de TODOS los pacientes procesados antes de subir a Google Cloud.
Detecta y reporta:
  1. Archivos NIfTI corruptos o ilegibles
  2. Dimensiones inconsistentes entre CT y máscara
  3. Máscaras completamente vacías (sin hueso)
  4. Tomografías con valores fuera de rango Hounsfield (-1100 a 3200 HU)
  5. Inconsistencias de tamaño (CT demasiado pequeño para el parche 192³)

Genera un reporte 'validation_report.json' con el listado de pacientes OK y los problemáticos.
"""

import os
import json
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
from datetime import datetime

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
PROCESSED_DIR   = "data/05_totalsegmentator/processed"
REPORT_OUT_PATH = "data/05_totalsegmentator/validation_report.json"
MIN_PATCH_SIZE  = 128    # Tamaño mínimo en cada dimensión para ser útil

# Rango válido de Unidades Hounsfield (HU) en tomografías
HU_MIN = -1100.0
HU_MAX =  3200.0

# =====================================================================


def validate_patient(patient_dir: str) -> dict:
    """
    Valida un paciente individual. Retorna un dict con el resultado.
    """
    patient_id = os.path.basename(patient_dir)
    result = {
        "id":     patient_id,
        "status": "OK",
        "issues": [],
        "stats":  {}
    }
    
    ct_path   = os.path.join(patient_dir, "ct.nii.gz")
    mask_path = os.path.join(patient_dir, "bone_mask.nii.gz")
    
    # --- Chequeo 1: Existencia de archivos ---
    if not os.path.exists(ct_path):
        result["issues"].append("MISSING: ct.nii.gz no encontrado")
        result["status"] = "ERROR"
        return result
    
    if not os.path.exists(mask_path):
        result["issues"].append("MISSING: bone_mask.nii.gz no encontrado")
        result["status"] = "ERROR"
        return result
    
    # --- Chequeo 2: Legibilidad de archivos ---
    try:
        ct_img   = nib.load(ct_path)
        ct_data  = ct_img.get_fdata()
    except Exception as e:
        result["issues"].append(f"CORRUPT: ct.nii.gz ilegible - {e}")
        result["status"] = "ERROR"
        return result
    
    try:
        mask_img  = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
    except Exception as e:
        result["issues"].append(f"CORRUPT: bone_mask.nii.gz ilegible - {e}")
        result["status"] = "ERROR"
        return result
    
    # --- Chequeo 3: Consistencia de dimensiones ---
    if ct_data.shape != mask_data.shape:
        result["issues"].append(
            f"SHAPE_MISMATCH: CT={ct_data.shape} vs Mask={mask_data.shape}"
        )
        result["status"] = "WARNING"
    
    # --- Chequeo 4: Tamaño mínimo para el parche 192³ ---
    min_dim = min(ct_data.shape)
    result["stats"]["shape"]   = list(ct_data.shape)
    result["stats"]["min_dim"] = int(min_dim)
    
    if min_dim < MIN_PATCH_SIZE:
        result["issues"].append(
            f"TOO_SMALL: Dimensión mínima={min_dim} < {MIN_PATCH_SIZE} (mínimo útil)"
        )
        result["status"] = "WARNING"
    
    # --- Chequeo 5: Rango Hounsfield ---
    ct_min = float(ct_data.min())
    ct_max = float(ct_data.max())
    result["stats"]["hu_min"] = round(ct_min, 2)
    result["stats"]["hu_max"] = round(ct_max, 2)
    
    if ct_min < HU_MIN - 200 or ct_max > HU_MAX + 200:
        result["issues"].append(
            f"HU_RANGE: Valores fuera de rango ({ct_min:.0f} a {ct_max:.0f} HU)"
        )
        result["status"] = "WARNING"
    
    # --- Chequeo 6: Máscara vacía ---
    bone_voxels = int(np.sum(mask_data > 0))
    result["stats"]["bone_voxels"] = bone_voxels
    result["stats"]["bone_ratio"]  = round(bone_voxels / mask_data.size, 6)
    
    if bone_voxels == 0:
        result["issues"].append("EMPTY_MASK: La máscara no contiene ningún vóxel de hueso")
        result["status"] = "ERROR"
    elif bone_voxels < 1000:
        result["issues"].append(
            f"SPARSE_MASK: Muy pocos vóxeles de hueso ({bone_voxels}). Posible error de segmentación."
        )
        result["status"] = "WARNING"
    
    return result


def main():
    print("=" * 60)
    print(" VALIDACIÓN DE INTEGRIDAD DEL DATASET V4")
    print("=" * 60)
    
    patient_dirs = sorted(glob.glob(os.path.join(PROCESSED_DIR, "s*")))
    
    if not patient_dirs:
        raise FileNotFoundError(f"No se encontraron pacientes en '{PROCESSED_DIR}'.")
    
    print(f"[*] Validando {len(patient_dirs)} pacientes...\n")
    
    results     = []
    ok_count    = 0
    warn_count  = 0
    error_count = 0
    
    for p_dir in tqdm(patient_dirs, desc="Validando"):
        try:
            res = validate_patient(p_dir)
        except Exception as e:
            res = {
                "id":     os.path.basename(p_dir),
                "status": "ERROR",
                "issues": [f"EXCEPTION: {e}"],
                "stats":  {}
            }
        
        results.append(res)
        
        if res["status"] == "OK":
            ok_count += 1
        elif res["status"] == "WARNING":
            warn_count += 1
        else:
            error_count += 1
    
    # --- Reporte final ---
    print(f"\n{'='*60}")
    print(f" RESULTADOS")
    print(f"{'='*60}")
    print(f"  ✅ OK:       {ok_count:>4} pacientes")
    print(f"  ⚠️  WARNINGS: {warn_count:>4} pacientes (usar con precaución)")
    print(f"  ❌ ERRORS:   {error_count:>4} pacientes (excluir del entrenamiento)")
    
    # Listar los problemáticos
    if warn_count + error_count > 0:
        print(f"\n[!] Pacientes con problemas:")
        for r in results:
            if r["status"] != "OK":
                print(f"  [{r['status']}] {r['id']}: {'; '.join(r['issues'])}")
    
    # Listas limpias para el split
    valid_patients = [r["id"] for r in results if r["status"] in ("OK", "WARNING")]
    error_patients = [r["id"] for r in results if r["status"] == "ERROR"]
    
    report = {
        "metadata": {
            "created_at":    datetime.now().isoformat(),
            "total":         len(patient_dirs),
            "ok":            ok_count,
            "warnings":      warn_count,
            "errors":        error_count,
            "usable":        len(valid_patients),
        },
        "valid_patients": valid_patients,
        "error_patients": error_patients,
        "detailed":       results
    }
    
    os.makedirs(os.path.dirname(REPORT_OUT_PATH), exist_ok=True)
    with open(REPORT_OUT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[✓] Reporte completo guardado en: {REPORT_OUT_PATH}")
    print(f"[*] Pacientes utilizables para entrenamiento: {len(valid_patients)}/{len(patient_dirs)}")


if __name__ == "__main__":
    main()
