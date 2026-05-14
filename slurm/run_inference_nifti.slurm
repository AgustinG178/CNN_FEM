#!/bin/bash
#SBATCH --job-name=Inference_V3_ep10
#SBATCH --partition=internos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=inference_v3_ep10_%j.log
#SBATCH --error=errores_inference_v3_ep10_%j.err

echo "============================================================"
echo " INFERENCIA V3 - ÉPOCA 10 - PACIENTE DE PRUEBA"
echo "============================================================"
echo "Nodo: $HOSTNAME"

export PYTHONPATH="$(pwd)"

# Tomamos el primer paciente disponible del dataset procesado como prueba
# (cualquier s000X sirve — el modelo nunca los vio si no estan en train)
PATIENT_DIR="data/05_totalsegmentator/processed/s0001"
CHECKPOINT="data/03_models/unet_v3_ep10.pth"
OUTPUT_DIR="data/02_processed/test_nifti_ep10"

# Verificar que existe el checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "[!] ERROR: No se encontró $CHECKPOINT"
    echo "    Checkpoints disponibles:"
    ls data/03_models/*.pth 2>/dev/null | head -20
    exit 1
fi

# Verificar que existe el paciente
if [ ! -d "$PATIENT_DIR" ]; then
    echo "[!] s0001 no encontrado, buscando el primero disponible..."
    PATIENT_DIR=$(ls -d data/05_totalsegmentator/processed/s* 2>/dev/null | head -1)
    echo "    Usando: $PATIENT_DIR"
fi

echo "Paciente: $PATIENT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo ""

python3 scripts/test_inference_nifti.py \
    --patient     "$PATIENT_DIR" \
    --checkpoint  "$CHECKPOINT"  \
    --output-dir  "$OUTPUT_DIR"  \
    --patch-size  128            \
    --overlap     0.5            \
    --threshold   0.5

echo ""
echo "Listo. Resultados en: $OUTPUT_DIR"
