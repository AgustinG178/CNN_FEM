#!/bin/bash
#SBATCH --job-name=boneflow_v3.2_wsd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=120:00:00
#SBATCH --output=experiments/v3.2_AdamW_WSD/v3.2_training_%j.log
#SBATCH --error=experiments/v3.2_AdamW_WSD/v3.2_training_%j.err

echo "============================================================"
echo " BONEFLOW AI - EXPERIMENTO V3.2 (ADAMW + WSD)"
echo " Nodo: $(hostname)"
echo " Inicio: $(date)"
echo "============================================================"

export PYTHONPATH=$PYTHONPATH:$(pwd)

python3 experiments/v3.2_AdamW_WSD/train_v32_wsd.py

echo "============================================================"
echo " Finalización: $(date)"
echo "============================================================"
