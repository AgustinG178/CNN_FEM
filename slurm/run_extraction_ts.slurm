#!/bin/bash
#SBATCH --job-name=TS_Extraction
#SBATCH --partition=internos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=extraccion_ts_%j.log 
#SBATCH --error=errores_extraccion_ts_%j.err        
#SBATCH --mail-user=agugrio17@gmail.com 
#SBATCH --mail-type=BEGIN,END,FAIL              
#SBATCH --no-requeue                      

echo "Iniciando procesamiento masivo de TotalSegmentator en el nodo: $HOSTNAME"

# Asegurar que nibabel esté instalado
pip install --user nibabel tqdm

export PYTHONPATH="$(pwd)"
python3 src/totalsegmentator_utils/extract_bones.py

echo "Extracción y fusión anatómica finalizada."
