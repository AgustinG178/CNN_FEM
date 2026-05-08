#!/bin/bash
#SBATCH --job-name=Prep_BoneFlow
#SBATCH --partition=internos              # FIUNER
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                # 32 cores para multiprocesamiento rápido
#SBATCH --mem=32G                         # 32GB de RAM (Límite seguro de la partición)
#SBATCH --output=prep_dataset_%j.log 
#SBATCH --error=prep_errores_%j.err        
#SBATCH --mail-user=agugrio17@gmail.com 
#SBATCH --mail-type=BEGIN,END,FAIL              
#SBATCH --no-requeue                      

# 1. Preparar el entorno
echo "====================================================="
echo " Iniciando GENERACION MASIVA DE DATASET V2 (128^3)"
echo " Nodo Asignado: $HOSTNAME"
echo "====================================================="

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 2. Limpieza preventiva (Forzando permisos)
echo "-> Forzando permisos y limpiando parches obsoletos..."
chmod -R 777 data/04_training_patches 2>/dev/null || true
rm -rf data/04_training_patches/*

# 3. Comandos de ejecución
echo "-> Lanzando prepare_dataset.py..."
export PYTHONPATH="$(pwd)"
python3 prepare_dataset.py

echo "-> ¡Generación de Dataset Finalizada!"
echo "-> Ya puedes lanzar run_cluster.slurm para iniciar el entrenamiento V2."
