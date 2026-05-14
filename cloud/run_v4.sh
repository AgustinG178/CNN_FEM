#!/bin/bash
# ============================================================
#  BONEFLOW V4: Crear VM en Google Cloud y lanzar entrenamiento
#  Ejecutar en tu PC local (PowerShell o bash con gcloud instalado)
# ============================================================

# CONFIGURACIÓN - EDITÁ ESTAS VARIABLES
PROJECT_ID="TU_PROJECT_ID"        # <-- Cambiar por tu Project ID de Google Cloud
BUCKET_NAME="boneflow-v4-data"    # <-- El mismo bucket que usaste en upload_to_gcs.sh
ZONE="us-central1-a"              # Zona con L4 disponible
VM_NAME="boneflow-v4"

echo "================================================"
echo " BoneFlow V4 - Creando VM en Google Cloud"
echo "================================================"

# Paso 1: Crear la VM con GPU L4
echo "[1/4] Creando instancia con GPU NVIDIA L4..."
gcloud compute instances create $VM_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --scopes=storage-full \
    --metadata="install-nvidia-driver=True" \
    --no-restart-on-failure

echo "[✓] VM creada. Esperando que inicie (60 segundos)..."
sleep 60

# Paso 2: Copiar el código fuente a la VM
echo "[2/4] Subiendo código fuente BoneFlow a la VM..."
gcloud compute scp --recurse \
    src/ \
    $VM_NAME:~/BoneFlow/src/ \
    --zone=$ZONE

# Paso 3: Ejecutar setup en la VM
echo "[3/4] Configurando el entorno en la VM..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    pip install torchio nibabel matplotlib --quiet
    mkdir -p ~/BoneFlow/data/03_models
    mkdir -p ~/BoneFlow/data/05_totalsegmentator
    
    echo '[*] Descargando datos de GCS...'
    gsutil -m cp -r gs://${BUCKET_NAME}/processed/ ~/BoneFlow/data/05_totalsegmentator/
    
    echo '[✓] Setup completo. Datos listos.'
    nvidia-smi
"

# Paso 4: Lanzar entrenamiento en background (nohup para que siga aunque se cierre SSH)
echo "[4/4] Lanzando entrenamiento V4..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="
    cd ~/BoneFlow
    nohup python3 src/neural_manifold/train_unet_v4.py \
        --patch_size 128 \
        --epochs 15 \
        --batch_size 2 \
        --max_lr 5e-4 \
        > training_v4.log 2>&1 &
    
    echo '[✓] Entrenamiento lanzado en background. PID: '$!
    echo 'Para monitorear: tail -f ~/BoneFlow/training_v4.log'
"

echo ""
echo "================================================"
echo " V4 INICIADO EN GOOGLE CLOUD 🚀"
echo "================================================"
echo " Para ver el log en tiempo real:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "   tail -f ~/BoneFlow/training_v4.log"
echo ""
echo " Para bajar el modelo final cuando termine:"
echo "   gcloud compute scp $VM_NAME:~/BoneFlow/data/03_models/unet_v4_BEST.pth ./"
echo ""
echo " Para APAGAR la VM cuando termine (¡importante para no gastar créditos!):"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo "================================================"
