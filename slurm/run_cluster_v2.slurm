#!/bin/bash
#SBATCH --job-name=UNet_V2_Original
#SBATCH --partition=internos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=training_v2_%j.log
#SBATCH --error=error_v2_%j.err

echo "Reanudando Entrenamiento V2 en el nodo: $HOSTNAME"

export PYTHONPATH="$(pwd)"
python3 src/neural_manifold/train_unet.py
