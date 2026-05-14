#!/bin/bash
#SBATCH --job-name=Prepare_V4_Dataset
#SBATCH --partition=internos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=prepare_v4_%j.log
#SBATCH --error=errores_prepare_v4_%j.err
#SBATCH --mail-user=agugrio17@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Iniciando preparación del Dataset V4 en el nodo: $HOSTNAME"
echo "Paso 1/2: Validando integridad de los 1228 pacientes..."

export PYTHONPATH="$(pwd)"

python3 -m pip install --user nibabel tqdm

# PASO 1: Validar integridad de cada NIfTI
python3 src/totalsegmentator_utils/validate_dataset.py

echo "Paso 2/2: Generando split estratificado Train/Val/Test..."

# PASO 2: Generar el split reproducible
python3 src/totalsegmentator_utils/split_dataset.py

echo ""
echo "Dataset V4 preparado. Archivos generados:"
echo "  - data/05_totalsegmentator/validation_report.json"
echo "  - data/05_totalsegmentator/dataset_split.json"
echo ""
echo "Siguiente paso: subir a Google Cloud con upload_to_gcs.py"
