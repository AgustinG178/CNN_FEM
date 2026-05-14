"""
upload_to_gcs.py
----------------
Sube el dataset procesado de TotalSegmentator a Google Cloud Storage.
Lee el archivo 'validation_report.json' para subir SOLO los pacientes válidos.

Prerrequisitos:
  pip install google-cloud-storage tqdm
  gcloud auth application-default login

Uso:
  python3 src/totalsegmentator_utils/upload_to_gcs.py \
      --bucket  tu-bucket-boneflow \
      --prefix  datasets/v4/processed \
      --dry-run          # Para simular sin subir

Estructura en GCS resultante:
  gs://tu-bucket-boneflow/
    datasets/v4/
      processed/
        s0001/
          ct.nii.gz
          bone_mask.nii.gz
        s0002/
          ...
      dataset_split.json
      validation_report.json
"""

import os
import json
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("[!] google-cloud-storage no instalado. Corré: pip install google-cloud-storage")


def get_gcs_client():
    """Crea el cliente de Google Cloud Storage."""
    if not GCS_AVAILABLE:
        raise ImportError("Instalá: pip install google-cloud-storage")
    return storage.Client()


def upload_file(bucket, local_path: str, gcs_path: str, dry_run: bool = False) -> bool:
    """
    Sube un archivo individual a GCS.
    Retorna True si se subió (o si dry_run), False si falló.
    """
    if dry_run:
        print(f"  [DRY-RUN] {local_path} -> gs://{bucket.name}/{gcs_path}")
        return True
    
    try:
        blob = bucket.blob(gcs_path)
        
        # Si el archivo ya existe con el mismo tamaño, lo saltea (resume logic)
        if blob.exists():
            blob.reload()
            local_size = os.path.getsize(local_path)
            if blob.size == local_size:
                return True  # Ya estaba subido
        
        blob.upload_from_filename(local_path)
        return True
    except Exception as e:
        print(f"\n  [ERROR] Al subir {local_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Sube el dataset BoneFlow a Google Cloud Storage")
    parser.add_argument("--bucket",  required=True, help="Nombre del bucket GCS (sin gs://)")
    parser.add_argument("--prefix",  default="datasets/v4", help="Prefijo en el bucket")
    parser.add_argument("--dry-run", action="store_true", help="Simula sin subir nada")
    parser.add_argument("--processed-dir", default="data/05_totalsegmentator/processed")
    parser.add_argument("--report",  default="data/05_totalsegmentator/validation_report.json")
    parser.add_argument("--split",   default="data/05_totalsegmentator/dataset_split.json")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f" SUBIENDO DATASET A GOOGLE CLOUD STORAGE")
    print(f" Destino: gs://{args.bucket}/{args.prefix}/")
    if args.dry_run:
        print(" ⚠️  MODO DRY-RUN: No se subirá nada")
    print("=" * 60)
    
    # 1. Leer la lista de pacientes válidos del reporte
    if not os.path.exists(args.report):
        raise FileNotFoundError(
            f"No se encontró '{args.report}'.\n"
            f"Ejecutá primero: python3 src/totalsegmentator_utils/validate_dataset.py"
        )
    
    with open(args.report) as f:
        report = json.load(f)
    
    valid_patients = set(report["valid_patients"])
    print(f"[*] Pacientes válidos a subir: {len(valid_patients)}")
    
    # 2. Listar archivos a subir
    patient_dirs = [
        d for d in sorted(glob.glob(os.path.join(args.processed_dir, "s*")))
        if os.path.basename(d) in valid_patients
    ]
    
    files_to_upload = []
    for p_dir in patient_dirs:
        patient_id = os.path.basename(p_dir)
        for filename in ["ct.nii.gz", "bone_mask.nii.gz"]:
            local_path = os.path.join(p_dir, filename)
            if os.path.exists(local_path):
                gcs_path = f"{args.prefix}/processed/{patient_id}/{filename}"
                files_to_upload.append((local_path, gcs_path))
    
    # También subir los archivos de metadata
    for meta_file in [args.report, args.split]:
        if os.path.exists(meta_file):
            gcs_path = f"{args.prefix}/{os.path.basename(meta_file)}"
            files_to_upload.append((meta_file, gcs_path))
    
    total_size_gb = sum(os.path.getsize(f[0]) for f in files_to_upload) / 1e9
    print(f"[*] Archivos a subir: {len(files_to_upload)} ({total_size_gb:.2f} GB)")
    
    if args.dry_run:
        print("\n[DRY-RUN] Primeros 10 archivos que se subirían:")
        for local, gcs in files_to_upload[:10]:
            size_mb = os.path.getsize(local) / 1e6
            print(f"  {local} ({size_mb:.1f}MB) -> gs://{args.bucket}/{gcs}")
        print(f"  ... y {len(files_to_upload)-10} más")
        return
    
    # 3. Conectar a GCS y subir
    client = get_gcs_client()
    bucket = client.bucket(args.bucket)
    
    # Verificar que el bucket existe
    if not bucket.exists():
        print(f"\n[ERROR] El bucket '{args.bucket}' no existe.")
        print(f"Crealo con: gsutil mb -l us-central1 gs://{args.bucket}")
        return
    
    print(f"\n[*] Iniciando subida... (esto puede tardar 30-60 minutos según tu conexión)")
    
    ok_count    = 0
    error_count = 0
    
    for local_path, gcs_path in tqdm(files_to_upload, desc="Subiendo"):
        success = upload_file(bucket, local_path, gcs_path, dry_run=args.dry_run)
        if success:
            ok_count += 1
        else:
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f" SUBIDA COMPLETADA")
    print(f"{'='*60}")
    print(f"  ✅ Subidos exitosamente: {ok_count}")
    print(f"  ❌ Errores:              {error_count}")
    print(f"\n  Dataset disponible en: gs://{args.bucket}/{args.prefix}/")
    print(f"\n  Para verificar en la consola:")
    print(f"  gsutil ls gs://{args.bucket}/{args.prefix}/processed/ | wc -l")


if __name__ == "__main__":
    main()
