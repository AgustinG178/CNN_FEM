#!/bin/bash
#SBATCH --job-name=boneflow_v3.1_sam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=experiments/v3.1_SAM_OneCycle/v3.1_training_%j.log
#SBATCH --error=experiments/v3.1_SAM_OneCycle/v3.1_training_%j.err

echo "============================================================"
echo " BONEFLOW AI - EXPERIMENTO V3.1 (SAM + ONECYCLE)"
echo " Nodo: $(hostname)"
echo " Inicio: $(date)"
echo "============================================================"

# Cargar entorno (ajustar si usas conda o venv)
# source activate boneflow_env

# PYTHONPATH al root para que los imports de 'src' funcionen
export PYTHONPATH=$PYTHONPATH:$(pwd)

python experiments/v3.1_SAM_OneCycle/train_v31_sam.py

echo "============================================================"
echo " Finalización: $(date)"
echo "============================================================"
