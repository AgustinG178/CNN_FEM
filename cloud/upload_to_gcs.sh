#!/bin/bash
# ============================================================
#  BONEFLOW V4: Subida de datos del CLÚSTER a Google Cloud Storage
#  Ejecutar en Putty/SSH en el clúster (nodo11)
# ============================================================

# INSTRUCCIONES:
# 1. Editá la variable BUCKET_NAME con el nombre de tu bucket de GCS
# 2. Ejecutá: bash cloud/upload_to_gcs.sh
# ============================================================

BUCKET_NAME="boneflow-v4-data"             # <-- CAMBIAR si usaste otro nombre
PROCESSED_DIR="$HOME/Automatizacion_FEM/data/05_totalsegmentator/processed"
GCS_DEST="gs://${BUCKET_NAME}/processed/"

echo "================================================"
echo " BoneFlow V4 - Subida de datos a Google Cloud"
echo "================================================"
echo "[*] Origen: $PROCESSED_DIR"
echo "[*] Destino: $GCS_DEST"

# Paso 1: Instalar rclone (binario único, sin root)
if ! command -v rclone &> /dev/null; then
    echo "[*] Instalando rclone (binario portátil, sin root)..."
    curl https://rclone.org/install.sh | bash --install-path="$HOME/.local/bin"
    export PATH="$HOME/.local/bin:$PATH"
fi

# Paso 2: Verificar que rclone está disponible
rclone --version || { echo "[ERROR] rclone no disponible. Instalación fallida."; exit 1; }

echo ""
echo "[!] CONFIGURACIÓN REQUERIDA:"
echo "    Necesitás un archivo de clave de cuenta de servicio JSON de Google Cloud."
echo "    Descargalo desde: Google Cloud Console > IAM > Cuentas de Servicio"
echo "    Guardalo en el clúster como: $HOME/gcs_key.json"
echo ""

KEY_FILE="$HOME/gcs_key.json"
if [ ! -f "$KEY_FILE" ]; then
    echo "[ERROR] No se encontró el archivo de clave: $KEY_FILE"
    echo "        Descargalo de Google Cloud y subilo por WinSCP."
    exit 1
fi

# Paso 3: Configurar rclone con la clave de servicio
RCLONE_CONFIG="$HOME/.config/rclone/rclone.conf"
mkdir -p "$(dirname "$RCLONE_CONFIG")"

cat > "$RCLONE_CONFIG" << EOF
[gcs]
type = google cloud storage
service_account_file = ${KEY_FILE}
EOF

echo "[✓] rclone configurado."

# Paso 4: Subir los datos (con resume automático si se corta)
echo "[*] Iniciando transferencia de 1228 pacientes a GCS..."
echo "    (Esto puede tardar 1-3 horas dependiendo de la velocidad de red)"

rclone copy "$PROCESSED_DIR" "gcs:${BUCKET_NAME}/processed/" \
    --progress \
    --transfers=8 \
    --checkers=16 \
    --stats=30s \
    --log-level INFO

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓] ¡Transferencia completada exitosamente!"
    echo "[✓] Datos disponibles en: $GCS_DEST"
    echo ""
    echo "Próximo paso: Crear la VM en Google Cloud y lanzar el entrenamiento."
    echo "Ejecutá en tu PC local: bash cloud/run_v4.sh"
else
    echo "[ERROR] La transferencia falló. Podés reintentar el mismo comando (rclone resume automáticamente)."
    exit 1
fi
