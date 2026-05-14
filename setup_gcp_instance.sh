#!/bin/bash
# =============================================================
# setup_gcp_instance.sh
# Script de configuración del entorno en la instancia de GCP.
# Correr UNA SOLA VEZ después de crear la VM.
#
# Uso:
#   chmod +x setup_gcp_instance.sh
#   ./setup_gcp_instance.sh
# =============================================================

set -e  # Detener si algo falla

echo "============================================================"
echo " CONFIGURANDO ENTORNO BONEFLOW V4 EN GOOGLE CLOUD"
echo "============================================================"

# 1. Actualizar el sistema
echo "[1/7] Actualizando paquetes del sistema..."
sudo apt-get update -q
sudo apt-get install -y -q unzip htop nvtop screen

# 2. Verificar GPU
echo "[2/7] Verificando GPU..."
nvidia-smi
python3 -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# 3. Instalar dependencias Python
echo "[3/7] Instalando librerías Python..."
pip install --upgrade pip
pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    torchio \
    nibabel \
    tqdm \
    matplotlib \
    google-cloud-storage \
    SimpleITK

# 4. Verificar PyTorch con CUDA
echo "[4/7] Verificando PyTorch + CUDA..."
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

# 5. Descargar el dataset desde GCS
echo "[5/7] Descargando dataset desde Google Cloud Storage..."
echo "      (Editá BUCKET_NAME abajo con el nombre de tu bucket)"
BUCKET_NAME="boneflow-v4"     # <-- CAMBIAR POR TU BUCKET
GCS_PREFIX="datasets/v4"

mkdir -p data/05_totalsegmentator/processed
mkdir -p data/03_models

# Descargar metadata primero
gsutil cp gs://$BUCKET_NAME/$GCS_PREFIX/dataset_split.json data/05_totalsegmentator/
gsutil cp gs://$BUCKET_NAME/$GCS_PREFIX/validation_report.json data/05_totalsegmentator/

# Descargar los pacientes (en paralelo con -m para máxima velocidad)
echo "      Descargando pacientes (esto tarda 10-20 min a velocidad GCS)..."
gsutil -m cp -r gs://$BUCKET_NAME/$GCS_PREFIX/processed/ data/05_totalsegmentator/

echo "      Dataset descargado. Verificando..."
PATIENT_COUNT=$(ls data/05_totalsegmentator/processed/ | wc -l)
echo "      Pacientes en disco: $PATIENT_COUNT"

# 6. Descargar el código del proyecto
echo "[6/7] Clonando el repositorio del proyecto..."
echo "      (Si no tenés repo Git, subí el código manualmente con gcloud scp)"
# git clone https://github.com/TU_USUARIO/boneflow.git .
# O si no tenés repo:
# gcloud compute scp --recurse ./src INSTANCE_NAME:~/boneflow/src

# 7. Verificar todo
echo "[7/7] Verificación final..."
python3 -c "
import torchio as tio
import nibabel as nib
import glob

patients = glob.glob('data/05_totalsegmentator/processed/s*')
print(f'✓ Pacientes en disco: {len(patients)}')
print(f'✓ torchio: {tio.__version__}')
print(f'✓ nibabel: {nib.__version__}')
print('✓ Entorno listo para entrenamiento V4')
"

echo ""
echo "============================================================"
echo " ENTORNO CONFIGURADO. Para lanzar V4:"
echo "   screen -S boneflow_v4"
echo "   python3 src/neural_manifold/train_unet_v4.py"
echo "   (Ctrl+A, D para dejar corriendo en background)"
echo "============================================================"
