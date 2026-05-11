#!/bin/bash
#SBATCH --job-name=Inference_BoneFlow
#SBATCH --partition=internos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=inferencia_%j.log
#SBATCH --error=error_inferencia_%j.err

echo "Lanzando inferencia en el nodo: $HOSTNAME"

export PYTHONPATH="$(pwd)"

# Ejemplo de uso: sbatch run_inference.slurm --epoch 80
python3 scripts/test_inference.py "$@"
