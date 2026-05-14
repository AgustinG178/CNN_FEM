#!/bin/bash
#SBATCH --job-name=SWA_V3.1
#SBATCH --partition=internos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=swa_v31_%j.log
#SBATCH --error=errores_swa_v31_%j.err
#SBATCH --mail-user=agugrio17@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Iniciando SWA (Stochastic Weight Averaging) para V3.1"
echo "Tomando los últimos 10 checkpoints de V3..."

export PYTHONPATH="$(pwd)"

# Evaluamos los últimos 10 checkpoints y promediamos los mejores 5
python3 src/neural_manifold/swa_ensemble.py \
    --checkpoints data/03_models/unet_v3_ep3*.pth data/03_models/unet_v3_ep4*.pth \
    --output      data/03_models/unet_v3_swa.pth \
    --val-split   data/05_totalsegmentator/dataset_split.json \
    --top-n       5 \
    --patch-size  128

echo "SWA V3.1 completado. Resultado en: data/03_models/unet_v3_swa.pth"
