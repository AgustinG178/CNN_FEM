#!/bin/bash
#SBATCH --job-name=CNN_FEM_Agus
#SBATCH --partition=internos              # FIUNER tiene prioridad absoluta aquí (6 nodos)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24                # 24 cores (Nodo completo según manual)
#SBATCH --mem=32G                         # 32GB de RAM (Seguro y suficiente)
#SBATCH --output=entrenamiento_%j.log 
#SBATCH --error=errores_%j.err        
#SBATCH --mail-user=agugrio17@gmail.com 
#SBATCH --mail-type=BEGIN,END,FAIL              
#SBATCH --no-requeue                      

# 1. Preparar el entorno
echo "Iniciando trabajo MASIVO de Deep Learning en el nodo: $HOSTNAME"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # Forzamos a PyTorch a usar los 24 cores para la matemática
mkdir -p logs # Crea la carpeta de logs por si no existe
mkdir -p data/03_models # Asegura que exista la carpeta para guardar los pesos finales

# Si en tu clúster usas módulos para Python/PyTorch, descomenta y edita esta línea:
# module load python/3.10 cuda/11.8

# Si usas un entorno virtual (recomendado si instalaste torchio y pytorch ahí):
# source .venv/bin/activate

# 2. Comandos de ejecución
echo "Lanzando UNet3D..."
export PYTHONPATH="$(pwd)"
python3 src/neural_manifold/train_unet.py

echo "Entrenamiento finalizado. Cerrando nodo."
