#!/bin/bash
#SBATCH --job-name=BoneFlow_V3
#SBATCH --partition=internos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=training_v3_%j.log 
#SBATCH --error=errores_training_v3_%j.err        
#SBATCH --mail-user=agugrio17@gmail.com 
#SBATCH --mail-type=BEGIN,END,FAIL              
#SBATCH --no-requeue                      

echo "Iniciando Entrenamiento Masivo V3 en el nodo: $HOSTNAME"
echo "Dataset: 1200 pacientes (TotalSegmentator) - Parcheo Dinámico en Memoria"

# Aseguramos que tenemos las librerías necesarias actualizadas
python3 -m pip install --user torchio matplotlib

export PYTHONPATH="$(pwd)"
python3 src/neural_manifold/train_unet_v3.py

echo "Entrenamiento V3 Finalizado."
