#!/bin/bash
#SBATCH --job-name=Batch_Inference_5pac
#SBATCH --partition=internos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=batch_inference_5pac_%j.log
#SBATCH --error=errores_batch_5pac_%j.err
#SBATCH --mail-user=agugrio17@gmail.com
#SBATCH --mail-type=END,FAIL

echo "============================================================"
echo " INFERENCIA EN BATCH — 5 PACIENTES ALEATORIOS — V3 EP10"
echo "============================================================"
echo "Nodo: $HOSTNAME"

export PYTHONPATH="$(pwd)"

CHECKPOINT="data/03_models/unet_v3_ep10.pth"
PROCESSED_DIR="data/05_totalsegmentator/processed"
OUTPUT_BASE="data/02_processed/batch_inference_ep10"

# Verificar checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "[!] ERROR: No se encontró $CHECKPOINT"
    exit 1
fi

# Seleccionar 5 pacientes aleatorios CON HUESO REAL desde el reporte de validación
echo "[*] Leyendo pacientes válidos desde validation_report.json..."
PATIENTS=($(python3 -c "
import json, random, os

report_path = 'data/05_totalsegmentator/validation_report.json'
if not os.path.exists(report_path):
    print('ERROR: no existe ' + report_path, file=__import__('sys').stderr)
    __import__('sys').exit(1)

with open(report_path) as f:
    report = json.load(f)

# Usar solo los pacientes marcados como válidos en el reporte
valid_ids = report['valid_patients']
valid_dirs = [f'data/05_totalsegmentator/processed/{pid}' for pid in valid_ids
              if os.path.isdir(f'data/05_totalsegmentator/processed/{pid}')]

random.shuffle(valid_dirs)
selected = valid_dirs[:5]
print(' '.join(selected))
"))

if [ $? -ne 0 ]; then
    echo "[!] ERROR al leer validation_report.json. ¿Está generado?"
    exit 1
fi
echo "[*] Pacientes seleccionados:"
for p in "${PATIENTS[@]}"; do echo "    $(basename $p)"; done
echo ""

# Variables de seguimiento
TOTAL=0
SUCCESS=0
TOTAL_DICE=0

for PATIENT_DIR in "${PATIENTS[@]}"; do
    PATIENT_ID=$(basename "$PATIENT_DIR")
    OUT_DIR="$OUTPUT_BASE/$PATIENT_ID"
    TOTAL=$((TOTAL + 1))

    echo "------------------------------------------------------------"
    echo "[$TOTAL/5] Procesando: $PATIENT_ID"
    echo "------------------------------------------------------------"

    # PASO 1: Inferencia + Dice Score
    python3 scripts/test_inference_nifti.py \
        --patient    "$PATIENT_DIR" \
        --checkpoint "$CHECKPOINT"  \
        --output-dir "$OUT_DIR"     \
        --patch-size 128            \
        --overlap    0.5            \
        --threshold  0.5

    # PASO 2: Generar STL de la máscara PREDICHA
    PRED_MASK="$OUT_DIR/${PATIENT_ID}_pred_mask.nii.gz"
    if [ -f "$PRED_MASK" ]; then
        echo "[*] Generando STL predicho..."
        python3 scripts/nifti_to_stl.py \
            --mask    "$PRED_MASK" \
            --output  "$OUT_DIR/${PATIENT_ID}_PREDICHO.stl" \
            --smooth  15
        SUCCESS=$((SUCCESS + 1))
    fi

    # PASO 3: Generar STL de la máscara de REFERENCIA (para comparación visual)
    REF_MASK="$PATIENT_DIR/bone_mask.nii.gz"
    if [ -f "$REF_MASK" ]; then
        echo "[*] Generando STL de referencia (ground truth)..."
        python3 scripts/nifti_to_stl.py \
            --mask    "$REF_MASK" \
            --output  "$OUT_DIR/${PATIENT_ID}_REFERENCIA.stl" \
            --smooth  15
    fi

    echo ""
done

echo "============================================================"
echo " RESUMEN"
echo "============================================================"
echo "  Pacientes procesados: $TOTAL"
echo "  STLs generados:       $SUCCESS"
echo "  Resultados en:        $OUTPUT_BASE"
echo ""
echo "  Para ver los STLs en 3D:"
echo "  Abrí los archivos .stl en MeshLab, Blender o 3D Slicer"
echo "  Comparar PREDICHO vs REFERENCIA para cada paciente"
echo "============================================================"
